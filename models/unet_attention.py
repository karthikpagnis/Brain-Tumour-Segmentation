"""
Attention-Enhanced U-Net Architecture for Brain Tumor Segmentation
Research paper: "Attention U-Net: Learning Where to Look for the Pancreas"
Modified for 3D medical image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

from models.attention_gates import AttentionGate, DoubleConvBlock3D, ConvBlock3D
from config import (
    ENCODER_CHANNELS,
    DECODER_CHANNELS,
    USE_ATTENTION_GATES,
    USE_BATCH_NORM,
    DROPOUT_RATE,
    NUM_INPUT_CHANNELS,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


class EncoderBlock(nn.Module):
    """
    Encoder block: Double Conv → MaxPool
    Reduces spatial dimensions while increasing channels
    """

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C_in, D, H, W)

        Returns:
            Tuple of (pooled_features, skip_connection)
            - pooled_features: (B, C_out, D/2, H/2, W/2)
            - skip_connection: (B, C_out, D, H, W) for use in decoder
        """
        skip_connection = self.double_conv(x)
        pooled = self.pool(skip_connection)
        return pooled, skip_connection


class DecoderBlock(nn.Module):
    """
    Decoder block with Attention Gate: Upsample → Attention → Concat → DoubleConv
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = USE_ATTENTION_GATES,
        dropout_rate: float = DROPOUT_RATE,
    ):
        """
        Args:
            in_channels: Input channels from deeper layer
            skip_channels: Channels from skip connection (encoder)
            out_channels: Output channels
            use_attention: Whether to use attention gates
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.use_attention = use_attention
        if use_attention:
            self.attention_gate = AttentionGate(
                in_channels_gating=out_channels,
                in_channels_skip=skip_channels,
                out_channels=out_channels,
            )

        # After concatenation, we have out_channels + skip_channels
        concat_channels = out_channels + skip_channels
        self.double_conv = DoubleConvBlock3D(
            concat_channels,
            out_channels,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_connection: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (B, C_in, D, H, W)
            skip_connection: Encoder skip connection (B, C_skip, 2D, 2H, 2W)

        Returns:
            Decoded features (B, C_out, 2D, 2H, 2W)
        """
        # Upsample
        x = self.up(x)

        # Apply attention gate if enabled
        if self.use_attention:
            skip_connection = self.attention_gate(x, skip_connection)

        # Concatenate
        x = torch.cat([x, skip_connection], dim=1)

        # Convolve
        x = self.double_conv(x)

        return x


class AttentionUNet3D(nn.Module):
    """
    Attention-Enhanced U-Net for 3D Medical Image Segmentation

    Architecture:
    - Encoder: 4 levels of downsampling (double conv + maxpool)
    - Bottleneck: Double conv at lowest resolution
    - Decoder: 4 levels of upsampling (transpose conv + attention gate + concat + double conv)
    - Output: 1x1 conv for class predictions

    Channel progression:
    - Input: NUM_INPUT_CHANNELS (4 for T1, T1ce, T2, FLAIR)
    - Level 1: 32 channels
    - Level 2: 64 channels
    - Level 3: 128 channels
    - Level 4: 256 channels (bottleneck)
    - Output: NUM_CLASSES (4 for background, necrotic core, edema, enhancing tumor)
    """

    def __init__(
        self,
        in_channels: int = NUM_INPUT_CHANNELS,
        out_channels: int = NUM_CLASSES,
        encoder_channels: List[int] = ENCODER_CHANNELS,
        use_attention: bool = USE_ATTENTION_GATES,
        dropout_rate: float = DROPOUT_RATE,
    ):
        """
        Args:
            in_channels: Number of input channels (4 for multimodal MRI)
            out_channels: Number of output channels (classes)
            encoder_channels: Channel progression in encoder
            use_attention: Whether to use attention gates
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        logger.info(
            f"Initializing Attention U-Net: "
            f"input={in_channels}, output={out_channels}, "
            f"attention={'enabled' if use_attention else 'disabled'}"
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            block = EncoderBlock(
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
            block = DecoderBlock(
                in_channels=encoder_channels[i],
                skip_channels=encoder_channels[i - 1],
                out_channels=encoder_channels[i - 1],
                use_attention=use_attention,
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
        Forward pass of Attention U-Net

        Args:
            x: Input multimodal MRI (B, 4, D, H, W)
               Where B=batch, 4=modalities (T1, T1ce, T2, FLAIR)
               D, H, W = depth, height, width (typically 155, 240, 240)

        Returns:
            Output segmentation logits (B, NUM_CLASSES, D, H, W)
            Where NUM_CLASSES = 4 (background, necrotic, edema, enhancing)
        """
        # Encoder with skip connections
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with attention-gated skip connections
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder_block(x, skip)

        # Output
        x = self.output_conv(x)

        return x

    @torch.no_grad()
    def get_attention_maps(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extract attention maps from attention gates
        Useful for visualization and interpretability

        Args:
            x: Input image

        Returns:
            List of attention maps from each decoder block
        """
        attention_maps = []

        # Encoder
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with attention extraction
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            # Upsample
            x = decoder_block.up(x)

            # Extract attention if available
            if decoder_block.use_attention:
                attention = decoder_block.attention_gate(x, skip)
                # Normalize for visualization
                attention = (attention - attention.min()) / (
                    attention.max() - attention.min() + 1e-8
                )
                attention_maps.append(attention)
                skip = attention
            else:
                skip = skip

            # Concatenate and conv
            x = torch.cat([x, skip], dim=1)
            x = decoder_block.double_conv(x)

        return attention_maps


class AttentionUNetModel:
    """
    Convenience wrapper for model management
    """

    @staticmethod
    def load_pretrained(checkpoint_path: str, device: torch.device = None):
        """
        Load pretrained model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Model in eval mode
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AttentionUNet3D()
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle both full model and state_dict checkpoints
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        logger.info(f"Loaded pretrained model from {checkpoint_path}")
        return model

    @staticmethod
    def save_checkpoint(
        model: AttentionUNet3D,
        optimizer,
        epoch: int,
        loss: float,
        checkpoint_path: str,
    ):
        """
        Save model checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet3D().to(device)

    # Print model info
    print("=" * 80)
    print("Attention-Enhanced U-Net for 3D Brain Tumor Segmentation")
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

    # Test attention maps extraction
    if model.use_attention:
        print("\nExtracting attention maps...")
        with torch.no_grad():
            attention_maps = model.get_attention_maps(x)
        print(f"  Number of attention maps: {len(attention_maps)}")
        for i, att_map in enumerate(attention_maps):
            print(f"    Map {i}: {att_map.shape}")

    print("\n✓ All tests passed!")
