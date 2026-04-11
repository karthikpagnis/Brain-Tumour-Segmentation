"""
Attention Gate Mechanisms for U-Net
Research-backed attention mechanisms to focus on tumor-relevant regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Recalibrate channel-wise feature responses

    Reference: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
    """

    def __init__(self, num_channels: int, reduction: int = 16):
        """
        Args:
            num_channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()
        reduced_channels = max(1, num_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, num_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) or (B, C, H, W)

        Returns:
            Attention-weighted features (same shape as input)
        """
        # Channel attention
        batch_size, num_channels = x.size(0), x.size(1)

        # Average pooling
        avg_out = self.avg_pool(x).view(batch_size, num_channels)
        avg_out = self.mlp(avg_out).view(batch_size, num_channels, 1, 1)
        if x.dim() == 5:
            avg_out = avg_out.unsqueeze(-1)

        # Max pooling
        max_out = self.max_pool(x).view(batch_size, num_channels)
        max_out = self.mlp(max_out).view(batch_size, num_channels, 1, 1)
        if x.dim() == 5:
            max_out = max_out.unsqueeze(-1)

        # Combine
        out = avg_out + max_out
        out = self.sigmoid(out)

        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Recalibrate spatial (2D/3D) feature responses
    """

    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Kernel size for convolution
        """
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or squeeze C dimension from (B, C, D, H, W)

        Returns:
            Attention-weighted features
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate
        x_out = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution followed by sigmoid
        out = self.sigmoid(self.conv(x_out))

        return x * out


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections

    Learns to suppress irrelevant features and amplify tumor-relevant regions
    Reference: "Attention U-Net: Learning Where to Look for the Pancreas"
               (Oktay et al., 2018)
    """

    def __init__(
        self,
        in_channels_gating: int,
        in_channels_skip: int,
        out_channels: int,
        sub_sample_factor: int = 2,
    ):
        """
        Args:
            in_channels_gating: Channels from gating signal (decoder path)
            in_channels_skip: Channels from skip connection (encoder path)
            out_channels: Output channels
            sub_sample_factor: Sub-sampling factor for resolution alignment
        """
        super().__init__()

        self.sub_sample_factor = sub_sample_factor

        # Channel alignment
        self.W_g = nn.Sequential(
            nn.Conv3d(
                in_channels_gating,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm3d(out_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(
                in_channels_skip,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm3d(out_channels),
        )

        # Gating
        self.psi = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        gating: torch.Tensor,
        skip_connection: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-gated skip connection

        Args:
            gating: Gating signal from decoder path (B, C_g, D, H, W)
            skip_connection: Skip connection from encoder (B, C_s, D, H, W)

        Returns:
            Attention-gated skip connection (B, C_s, D, H, W)
        """
        # Sub-sample gating signal if needed
        if self.sub_sample_factor > 1:
            gating = F.interpolate(
                gating,
                scale_factor=1.0 / self.sub_sample_factor,
                mode="nearest",
            )

        # Align channels
        g1 = self.W_g(gating)
        x1 = self.W_x(skip_connection)

        # Combine
        combination = self.relu(g1 + x1)

        # Compute attention weights
        psi = self.psi(combination)

        # Up-sample attention weights if originally sub-sampled
        if self.sub_sample_factor > 1:
            psi = F.interpolate(
                psi,
                size=skip_connection.shape[2:],
                mode="nearest",
            )

        # Apply attention to skip connection
        out = skip_connection * psi

        return out


class HybridAttention(nn.Module):
    """
    Hybrid Attention combining both channel and spatial attention
    Sequential application: Channel Attention → Spatial Attention
    """

    def __init__(self, num_channels: int, kernel_size: int = 7):
        """
        Args:
            num_channels: Number of input channels
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        self.channel_attention = ChannelAttention(num_channels)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel then spatial attention

        Args:
            x: Input features (B, C, H, W)

        Returns:
            Attention-weighted features
        """
        # Apply channel attention
        out = self.channel_attention(x)

        # Apply spatial attention
        out = self.spatial_attention(out)

        return out


class ConvBlock3D(nn.Module):
    """
    3D Convolutional Block: Conv → BatchNorm → ReLU → Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dropout_rate: Dropout probability
        """
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvBlock3D(nn.Module):
    """
    Double 3D Convolutional Block: (Conv → BN → ReLU) × 2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            ConvBlock3D(in_channels, out_channels, kernel_size, dropout_rate),
            ConvBlock3D(out_channels, out_channels, kernel_size, dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


if __name__ == "__main__":
    # Test attention mechanisms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 2, 64, 32, 32, 32

    # Create dummy input
    x = torch.randn(batch_size, channels, depth, height, width).to(device)

    # Test Channel Attention
    print("Testing Channel Attention...")
    channel_att = ChannelAttention(channels).to(device)
    out_channel = channel_att(x)
    print(f"  Input: {x.shape}, Output: {out_channel.shape}")

    # Test Attention Gate
    print("\nTesting Attention Gate...")
    att_gate = AttentionGate(
        in_channels_gating=128,
        in_channels_skip=64,
        out_channels=64,
    ).to(device)
    gating = torch.randn(batch_size, 128, 16, 16, 16).to(device)
    skip = torch.randn(batch_size, 64, 32, 32, 32).to(device)
    out_gate = att_gate(gating, skip)
    print(f"  Gating: {gating.shape}, Skip: {skip.shape}")
    print(f"  Output: {out_gate.shape}")

    # Test Double Conv Block
    print("\nTesting Double Conv Block...")
    double_conv = DoubleConvBlock3D(channels, channels * 2).to(device)
    out_conv = double_conv(x)
    print(f"  Input: {x.shape}, Output: {out_conv.shape}")

    print("\nAll tests passed!")
