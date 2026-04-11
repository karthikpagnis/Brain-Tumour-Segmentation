"""
Standard U-Net Baseline Model
For comparison with Attention-Enhanced U-Net
"""

import torch
import torch.nn as nn
from typing import List
import logging

from config import (
    ENCODER_CHANNELS,
    USE_BATCH_NORM,
    DROPOUT_RATE,
    NUM_INPUT_CHANNELS,
    NUM_CLASSES,
)
from models.attention_gates import DoubleConvBlock3D

logger = logging.getLogger(__name__)


class StandardEncoderBlock(nn.Module):
    """Encoder block: Double Conv → MaxPool (no attention)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = DROPOUT_RATE,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.double_conv = DoubleConvBlock3D(
            in_channels,
            out_channels,
            dropout_rate=dropout_rate,
        )

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip_connection = self.double_conv(x)
        pooled = self.pool(skip_connection)
        return pooled, skip_connection


class StandardDecoderBlock(nn.Module):
    """Decoder block: Upsample → Concat → DoubleConv (no attention)"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float = DROPOUT_RATE,
    ):
        """
        Args:
            in_channels: Input channels from deeper layer
            skip_channels: Channels from skip connection
            out_channels: Output channels
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        # After concatenation: out_channels + skip_channels
        concat_channels = out_channels + skip_channels
        self.double_conv = DoubleConvBlock3D(
            concat_channels,
            out_channels,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Decoder input (B, C_in, D, H, W)
            skip_connection: Encoder skip connection (B, C_skip, 2D, 2H, 2W)

        Returns:
            Decoded features (B, C_out, 2D, 2H, 2W)
        """
        # Upsample
        x = self.up(x)

        # Simple concatenation (no attention gate)
        x = torch.cat([x, skip_connection], dim=1)

        # Convolve
        x = self.double_conv(x)

        return x


class StandardUNet3D(nn.Module):
    """
    Standard U-Net for 3D Medical Image Segmentation
    Baseline model without attention mechanisms

    This model serves as the baseline for comparing with Attention-Enhanced U-Net.
    Architecture is identical except for the removal of attention gates.

    Identical to AttentionUNet3D except:
    - No attention gates on skip connections
    - Simpler decoder blocks
    """

    def __init__(
        self,
        in_channels: int = NUM_INPUT_CHANNELS,
        out_channels: int = NUM_CLASSES,
        encoder_channels: List[int] = ENCODER_CHANNELS,
        dropout_rate: float = DROPOUT_RATE,
    ):
        """
        Args:
            in_channels: Number of input channels (4 for multimodal MRI)
            out_channels: Number of output channels (classes)
            encoder_channels: Channel progression in encoder
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        logger.info(
            f"Initializing Standard U-Net: "
            f"input={in_channels}, output={out_channels}, "
            f"attention=disabled"
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            block = StandardEncoderBlock(
                encoder_channels[i],
                encoder_channels[i + 1],
                dropout_rate=dropout_rate,
            )
            self.encoder_blocks.append(block)

        # Bottleneck
        self.bottleneck = DoubleConvBlock3D(
            encoder_channels[-1],
            encoder_channels[-1],
            dropout_rate=dropout_rate,
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(encoder_channels) - 1, 0, -1):
            block = StandardDecoderBlock(
                in_channels=encoder_channels[i],
                skip_channels=encoder_channels[i - 1],
                out_channels=encoder_channels[i - 1],
                dropout_rate=dropout_rate,
            )
            self.decoder_blocks.append(block)

        # Output layer
        self.output_conv = nn.Conv3d(
            encoder_channels[0],
            out_channels,
            kernel_size=1,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Standard U-Net

        Args:
            x: Input multimodal MRI (B, 4, D, H, W)

        Returns:
            Output segmentation logits (B, NUM_CLASSES, D, H, W)
        """
        # Encoder with skip connections
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with simple skip concatenation
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder_block(x, skip)

        # Output
        x = self.output_conv(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models():
    """Compare Standard U-Net and Attention U-Net architectures"""
    from models.unet_attention import AttentionUNet3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create both models
    standard_unet = StandardUNet3D().to(device)
    attention_unet = AttentionUNet3D().to(device)

    # Count parameters
    standard_params = count_parameters(standard_unet)
    attention_params = count_parameters(attention_unet)

    print("=" * 80)
    print("Model Comparison: Standard U-Net vs Attention-Enhanced U-Net")
    print("=" * 80)

    print(f"\nStandard U-Net:")
    print(f"  Parameters: {standard_params:,}")

    print(f"\nAttention-Enhanced U-Net:")
    print(f"  Parameters: {attention_params:,}")

    print(f"\nDifference: {attention_params - standard_params:,} additional parameters")
    print(f"  ({100 * (attention_params - standard_params) / standard_params:.1f}% increase)")

    # Test forward pass
    print("\n" + "=" * 80)
    print("Forward Pass Test")
    print("=" * 80)

    batch_size = 1
    x = torch.randn(batch_size, NUM_INPUT_CHANNELS, 155, 240, 240).to(device)

    with torch.no_grad():
        standard_out = standard_unet(x)
        attention_out = attention_unet(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Standard U-Net output: {standard_out.shape}")
    print(f"Attention U-Net output: {attention_out.shape}")
    print(f"Expected: ({batch_size}, {NUM_CLASSES}, 155, 240, 240)")

    print("\n✓ Both models working correctly!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test standard U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandardUNet3D().to(device)

    print("=" * 80)
    print("Standard U-Net for 3D Brain Tumor Segmentation")
    print("=" * 80)
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Device: {device}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, NUM_INPUT_CHANNELS, 155, 240, 240).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {NUM_CLASSES}, 155, 240, 240)")

    # Compare with attention model
    print("\n")
    compare_models()
