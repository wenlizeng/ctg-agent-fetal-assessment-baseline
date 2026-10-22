# -*- coding: utf-8 -*-
"""
Task 2 — CTG Pattern Recognition: 1D U-Net (PyTorch).

Model: lightweight 1D U-Net with residual blocks for multi-class segmentation of
FHR patterns (Baseline / Acceleration / Deceleration / Other).

Reference: Liu et al., "Baseline/acceleration/deceleration determination
of fetal heart rate signals using a novel ensemble LCResU-Net,"
Expert Systems with Applications, 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Residual Conv Block
# ===========================================================================

class ResidualConvBlock(nn.Module):
    """1D residual convolution block: Conv→BN→ReLU→Dropout→Conv→BN→+→ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=7, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut projection if channel dimensions differ
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + shortcut
        out = self.relu(out)
        return out


# ===========================================================================
# 1D U-Net for CTG Segmentation
# ===========================================================================

class CTGSegUNet(nn.Module):
    """
    1D U-Net for CTG pattern segmentation.

    Parameters
    ----------
    in_channels : int — input channels (default 2: FHR + UC)
    num_classes : int — output classes (default 4)
    filters : list[int] — encoder filters per level
    dropout : float
    """

    def __init__(self, in_channels=1, num_classes=3,
                 filters=None, dropout=0.1):
        super().__init__()
        if filters is None:
            filters = [16, 32, 64, 128]

        self.num_classes = num_classes
        encoder_filters = filters[:-1]
        bottleneck_filters = filters[-1]

        # === Encoder ===
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for f in encoder_filters:
            self.enc_blocks.append(
                ResidualConvBlock(in_ch, f, kernel_size=5, dropout=dropout))
            self.pools.append(nn.MaxPool1d(kernel_size=2))
            in_ch = f

        # Bottleneck
        self.bottleneck = ResidualConvBlock(
            in_ch, bottleneck_filters, kernel_size=5, dropout=dropout)

        # === Decoder ===
        self.up_samples = nn.ModuleList()
        self.dec_adjust = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        decoder_filters = list(reversed(encoder_filters))
        in_ch = bottleneck_filters
        for f in decoder_filters:
            self.up_samples.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.dec_adjust.append(nn.Conv1d(in_ch, f, kernel_size=1))
            self.dec_blocks.append(
                ResidualConvBlock(f * 2, f, kernel_size=5, dropout=dropout))
            in_ch = f

        # Output
        self.out_conv = nn.Conv1d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor (B, C, T)

        Returns
        -------
        torch.Tensor (B, num_classes, T)
        """
        # Encoder
        skips = []
        for enc_block, pool in zip(self.enc_blocks, self.pools):
            x = enc_block(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, (up, adj, dec_block) in enumerate(
                zip(self.up_samples, self.dec_adjust, self.dec_blocks)):
            x = up(x)
            x = adj(x)
            skip = skips[-(i + 1)]

            # Handle odd-length padding mismatch
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    skip = F.pad(skip, (0, -diff))

            x = torch.cat([x, skip], dim=1)  # concat along channel dim
            x = dec_block(x)

        x = self.out_conv(x)
        return F.softmax(x, dim=1)


# ===========================================================================
# Combined Loss: Dice + Focal
# ===========================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """y_pred: (B, C, T), y_true: (B, C, T)"""
        intersection = (y_true * y_pred).sum(dim=[1, 2])
        union = y_true.sum(dim=[1, 2]) + y_pred.sum(dim=[1, 2])
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """y_pred: (B, C, T), y_true: (B, C, T)"""
        y_pred = y_pred.clamp(1e-7, 1.0)
        ce = -y_true * torch.log(y_pred)
        pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
        focal = self.alpha * torch.pow(1 - pt, self.gamma) * ce
        return focal.mean()


class CombinedSegmentationLoss(nn.Module):
    """Dice + Focal combined loss."""

    def __init__(self, focal_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.focal_weight = focal_weight

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + \
               self.focal_weight * self.focal(y_pred, y_true)


# ===========================================================================
# Model Factory
# ===========================================================================

def build_task2_model(in_channels=1, num_classes=3, filters=None,
                      dropout=0.1):
    """Build the official lightweight Task 2 1D U-Net baseline."""
    return CTGSegUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        filters=filters,
        dropout=dropout,
    )
