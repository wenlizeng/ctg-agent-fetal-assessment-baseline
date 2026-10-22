# -*- coding: utf-8 -*-
"""
Task 3 — Fetal Well-being Assessment: 1D-CNN Binary Classifier (PyTorch).

Classifies CTG recordings as Normal/Abnormal (FIGO) and predicts
neonatal asphyxia risk from FHR + UC waveforms.

Reference: JNU-CTG dataset baseline — 1D-CNN architecture.
"""

import torch
import torch.nn as nn


class Task3CNN(nn.Module):
    """
    1D-CNN for CTG classification.

    Architecture:
      Block 1: Conv1d(2→32, 15, stride 2) → BN → ReLU → MaxPool1d(2)
      Block 2: Conv1d(32→64, 7, stride 1)  → BN → ReLU → MaxPool1d(2)
      Block 3: Conv1d(64→128, 5, stride 1) → BN → ReLU → MaxPool1d(2)
      GlobalAveragePool1d → Linear(128→64) → Dropout → Linear(64→1) → Sigmoid

    Parameters
    ----------
    input_channels : int — number of input channels (default 2: FHR + UC)
    input_length : int — sequence length (default 7200)
    dropout : float
    """

    def __init__(self, input_channels=2, input_length=7200, dropout=0.5):
        super().__init__()

        # Block 1 — large receptive field
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Block 2 — medium receptive field
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Block 3 — fine receptive field
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Global pooling + classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor (B, C, T)

        Returns
        -------
        torch.Tensor (B, 1)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)  # (B, 128)
        x = self.classifier(x)        # (B, 1)
        return x


# ===========================================================================
# Task 3 with Task 2 Feature Fusion (for Task 4 pipeline)
# ===========================================================================

class Task3PatternFusedCNN(nn.Module):
    """
    Extended 1D-CNN that fuses waveform and pattern segmentation features.

    Two input branches:
      - Waveform branch: 1D-CNN over FHR
      - Pattern branch: 1D-CNN over segmentation probabilities
    Merged before final classifier.
    """

    def __init__(self, input_channels_wave=2, input_channels_pattern=4,
                 input_length=7200, dropout=0.5):
        super().__init__()

        # Waveform branch (same as Task3CNN minus classifier)
        self.wave_block1 = nn.Sequential(
            nn.Conv1d(input_channels_wave, 32, 15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.wave_block2 = nn.Sequential(
            nn.Conv1d(32, 64, 7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.wave_block3 = nn.Sequential(
            nn.Conv1d(64, 128, 5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.wave_gap = nn.AdaptiveAvgPool1d(1)

        # Pattern branch
        self.pat_block1 = nn.Sequential(
            nn.Conv1d(input_channels_pattern, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.pat_block2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.pat_gap = nn.AdaptiveAvgPool1d(1)

        # Fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, wave_input, pattern_input):
        """Forward pass with dual inputs.

        Parameters
        ----------
        wave_input : torch.Tensor (B, 1, T)
        pattern_input : torch.Tensor (B, 3, T)

        Returns
        -------
        torch.Tensor (B, 1)
        """
        # Waveform branch
        w = self.wave_block1(wave_input)
        w = self.wave_block2(w)
        w = self.wave_block3(w)
        w = self.wave_gap(w).squeeze(-1)  # (B, 128)

        # Pattern branch
        p = self.pat_block1(pattern_input)
        p = self.pat_block2(p)
        p = self.pat_gap(p).squeeze(-1)   # (B, 64)

        # Fusion
        fused = torch.cat([w, p], dim=1)  # (B, 192)
        return self.classifier(fused)


def build_task3_model(input_channels=2, input_length=7200, dropout=0.5):
    """Build and return the Task 3 1D-CNN model."""
    return Task3CNN(
        input_channels=input_channels,
        input_length=input_length,
        dropout=dropout,
    )


def build_task3_fused_model(input_channels_wave=1, input_channels_pattern=3,
                            input_length=4800, dropout=0.5):
    """Build the Task 3 classifier used by the staged Task 4 workflow."""
    return Task3PatternFusedCNN(
        input_channels_wave=input_channels_wave,
        input_channels_pattern=input_channels_pattern,
        input_length=input_length,
        dropout=dropout,
    )
