# -*- coding: utf-8 -*-
"""
AutoFHR: Neural Temporal Model for Fetal Heart Rate Estimation from DUS (PyTorch).

Architecture: dilated causal convolutions, residual blocks with gated
activations, multi-head attention, and combined BCE + spectral fidelity loss.

Reference: Rafiei et al., IEEE TBME, 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# CausalConv1d helper
# ===========================================================================

class CausalConv1d(nn.Module):
    """1D causal convolution using left-padding + standard Conv1d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, stride=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        # Pad on left side only for causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


# ===========================================================================
# AutoFHR Model
# ===========================================================================

class AutoFHRModel(nn.Module):
    """AutoFHR model for heartbeat localization in DUS scalograms.

    Parameters
    ----------
    input_shape : tuple — (time_bins, freq_bins), e.g. (1000, 40)
    filters : int — conv filters per block
    kernel_size : int — kernel width
    dilation_rates : list[int] or None
    num_heads : int — attention heads
    key_dim : int — attention key dimension
    dropout : float
    """

    def __init__(self, input_shape=(1000, 40), filters=40, kernel_size=20,
                 dilation_rates=None, num_heads=8, key_dim=40, dropout=0.25):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [2 ** i for i in range(8)]  # 1,2,4,...,128

        time_bins, freq_bins = input_shape
        self.dropout = nn.Dropout(dropout)

        # Gated activation blocks
        self.blocks = nn.ModuleList()
        for dr in dilation_rates:
            self.blocks.append(
                GatedActivationBlock(freq_bins, filters, kernel_size, dr))

        # Conv1x1 for skip projection
        self.skip_adjust = nn.ModuleList()
        for _ in dilation_rates:
            self.skip_adjust.append(
                nn.Conv1d(filters, filters, kernel_size=1, bias=True))

        # Post-processing layers
        self.dropout_layer = nn.Dropout(dropout)

        # Multi-head attention (operates on full feature dimension)
        self.attention = nn.MultiheadAttention(
            embed_dim=filters, num_heads=num_heads,
            batch_first=True)

        self.conv_out1 = nn.Conv1d(filters, 1, kernel_size=1, bias=True)
        self.conv_out2 = nn.Conv1d(1, 1, kernel_size=1, bias=True)

        # TimeDistributed Dense (1, sigmoid) → Conv1d(1, 1, 1) + Sigmoid
        self.final_conv = nn.Conv1d(1, 1, kernel_size=1)
        self.final_sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # Kaiming init: normal(0, 0.02) previously left every layer near
        # zero, which (together with chained sigmoids) froze gradients at
        # the pos_weight-balanced saddle point — train_loss stuck at the
        # all-zero-logit value. Kaiming keeps gradient scale healthy.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor (B, T, F) — scalogram

        Returns
        -------
        torch.Tensor (B, T, 1) — heartbeat probability mask
        """
        # TF model expects (B, T, F); PyTorch Conv1d expects (B, C, T)
        # Permute: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)  # (B, F, T)

        skip_connections = []
        inp = x  # for residual

        for i, block in enumerate(self.blocks):
            x, skip = block(x, inp)
            skip = self.skip_adjust[i](skip)
            skip_connections.append(skip)

        # Sum skip connections
        out = torch.stack(skip_connections, dim=0).sum(dim=0)  # (B, filters, T)
        out = self.dropout_layer(out)

        # Multi-head attention on full feature dimension: (B, filters, T) → (B, T, filters)
        attn_in = out.permute(0, 2, 1)  # (B, T, filters)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)  # (B, T, filters)
        attn_out = attn_out.permute(0, 2, 1)  # (B, filters, T)
        out = out + attn_out

        # Linear head → raw logits (BCEWithLogitsLoss). Intermediate
        # sigmoids were removed: chained sigmoid(conv(x)) layers multiplied
        # gradients by <=0.25 each and froze training at the zero-logit
        # saddle point.
        out = self.conv_out1(out)  # (B, 1, T)
        out = self.conv_out2(out)  # (B, 1, T)

        # TimeDistributed dense — output raw logits (BCEWithLogitsLoss)
        out = self.final_conv(out)  # (B, 1, T)

        # Back to (B, T, 1)
        out = out.permute(0, 2, 1)
        return out


class GatedActivationBlock(nn.Module):
    """Gated activation unit: tanh ⊙ sigmoid, with residual + skip."""

    def __init__(self, in_channels, filters, kernel_size, dilation_rate):
        super().__init__()
        self.tanh_conv = CausalConv1d(
            in_channels, filters, kernel_size, dilation=dilation_rate)
        self.sigmoid_conv = CausalConv1d(
            in_channels, filters, kernel_size, dilation=dilation_rate)
        self.skip_conv = nn.Conv1d(filters, filters, kernel_size=1, bias=True)
        self.skip_proj = nn.Conv1d(in_channels, filters, kernel_size=1, bias=True)

    def forward(self, x, input_layer):
        """
        Args:
            x: (B, C, T) current features
            input_layer: (B, C, T) original input for residual
        Returns:
            x: (B, filters, T) output features
            skip: (B, filters, T) skip connection
        """
        tanh_out = torch.tanh(self.tanh_conv(x))
        sigmoid_out = torch.sigmoid(self.sigmoid_conv(x))
        gated = tanh_out * sigmoid_out  # element-wise product

        skip = self.skip_conv(gated)

        # Project input_layer to match channels for residual
        projected_inp = self.skip_proj(input_layer)
        x = gated + projected_inp

        return x, skip


# ===========================================================================
# Custom Loss: BCE + Spectral Fidelity
# ===========================================================================

def autocorr(signal):
    """Compute autocorrelation via FFT (Wiener-Khinchin theorem).

    Much more memory-efficient than conv1d with large kernel.

    Args:
        signal: (B, T) tensor
    Returns:
        ac: (B, T) autocorrelation (unnormalised)
    """
    B, T = signal.shape
    signal = signal.float()
    mean = signal.mean(dim=1, keepdim=True)
    centered = signal - mean                        # (B, T)

    # FFT-based autocorrelation — O(T log T) instead of O(T²)
    n_fft = 2 * T
    fft = torch.fft.rfft(centered, n=n_fft, dim=1)  # (B, T+1)
    power = fft.abs().pow(2)                          # power spectrum
    ac_full = torch.fft.irfft(power, n=n_fft, dim=1) # (B, 2T)
    return ac_full[:, :T]                             # (B, T) — first T lags


class AutoFHRLoss(nn.Module):
    """Combined BCE-with-logits + autocorrelation mismatch penalty.

    Uses BCEWithLogitsLoss with pos_weight to combat extreme class imbalance
    (~2.7% positive bins in heartbeat masks).
    """

    def __init__(self, pos_weight=30.0, penalty_weight=0.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        # Register as buffer so pos_weight follows the model/device
        # (a plain CPU tensor previously caused a device-mismatch error
        # when training on CUDA)
        self.register_buffer(
            'pos_weight', torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, y_pred, y_true):
        """y_pred: (B, T, 1) raw logits, y_true: (B, T, 1) in [0, 1]"""
        y_pred = y_pred.squeeze(-1)  # (B, T) logits
        y_true = y_true.squeeze(-1)  # (B, T)

        # BCE with logits — pos_weight makes skipping positives expensive
        pos_weight = self.pos_weight.to(y_pred.device)
        bce = F.binary_cross_entropy_with_logits(
            y_pred, y_true.float(), pos_weight=pos_weight)

        if self.penalty_weight > 0:
            # Spectral penalty: convert logits → probabilities first
            y_prob = torch.sigmoid(y_pred).clamp(1e-7, 1 - 1e-7)
            y_true_clamped = y_true.float().clamp(1e-7, 1 - 1e-7)
            ac_true = autocorr(y_true_clamped)
            ac_pred = autocorr(y_prob)
            penalty = torch.mean((ac_true - ac_pred) ** 2)
            return bce + self.penalty_weight * penalty

        return bce


# ===========================================================================
# Model Factory
# ===========================================================================

def build_task1_model(input_shape=(1000, 40), cfg=None):
    """Factory function: build the Task 1 AutoFHR model."""
    if cfg is None:
        from config import TASK1_CFG as cfg

    model = AutoFHRModel(
        input_shape=input_shape,
        filters=cfg['filters'],
        kernel_size=cfg['kernel_size'],
        num_heads=cfg['num_heads'],
        key_dim=cfg['key_dim'],
        dropout=cfg.get('dropout_rate', 0.25),
    )
    return model
