"""
Main Training Loop for Brain Tumor Segmentation Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import sys

from config import *
from models.unet_attention import AttentionUNet3D
from models.loss_functions import DiceBCELoss
from data.dataloader import BraTS2021DataLoader
from training.metrics import MetricAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("outputs/training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for Attention U-Net"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        experiment_name: str = EXPERIMENT_NAME,
    ):
        """
        Initialize trainer

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            experiment_name: Name for experiment tracking
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.experiment_name = experiment_name

        # Loss function
        self.loss_fn = DiceBCELoss(
            dice_weight=DICE_LOSS_WEIGHT,
            bce_weight=BCE_LOSS_WEIGHT,
        ).to(device)

        # Optimizer
        if OPTIMIZER.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
        elif OPTIMIZER.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=LEARNING_RATE,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

        # Learning rate scheduler
        if SCHEDULER.lower() == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=SCHEDULER_FACTOR,
                patience=SCHEDULER_PATIENCE,
                verbose=True,
                min_lr=LEARNING_RATE_MIN,
            )
        elif SCHEDULER.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=NUM_EPOCHS,
                eta_min=LEARNING_RATE_MIN,
            )
        else:
            self.scheduler = None

        # TensorBoard
        self.writer = None
        if USE_TENSORBOARD:
            log_dir = TENSORBOARD_LOG_DIR / experiment_name
            self.writer = SummaryWriter(log_dir=log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.best_model_path = CHECKPOINTS_DIR / f"{experiment_name}_best.pth"
        self.patience_counter = 0

        logger.info(f"Trainer initialized for experiment: {experiment_name}")

    def train_epoch(self) -> dict:
        """
        Train for one epoch

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.current_epoch += 1

        metrics = MetricAggregator(NUM_CLASSES)
        total_loss = 0.0
        loss_components = {"dice": 0.0, "bce": 0.0}

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            targets = batch["segmentation"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)

            # Loss computation
            loss, components = self.loss_fn(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if GRADIENT_CLIP_VALUE > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    GRADIENT_CLIP_VALUE,
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            loss_components["dice"] += components["dice"]
            loss_components["bce"] += components["bce"]

            metrics.update(predictions.detach(), targets)

            # Print progress
            if (batch_idx + 1) % PRINT_FREQUENCY == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"Epoch {self.current_epoch}/{NUM_EPOCHS} "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} "
                    f"Loss: {avg_loss:.4f}"
                )

        # Average metrics
        epoch_metrics = metrics.get_scores()
        epoch_metrics["loss"] = total_loss / len(self.train_loader)
        epoch_metrics["loss_dice"] = loss_components["dice"] / len(self.train_loader)
        epoch_metrics["loss_bce"] = loss_components["bce"] / len(self.train_loader)

        return epoch_metrics

    def validate(self) -> dict:
        """
        Validate on validation set

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        metrics = MetricAggregator(NUM_CLASSES)
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                targets = batch["segmentation"].to(self.device)

                # Forward pass
                predictions = self.model(images)

                # Loss
                loss, _ = self.loss_fn(predictions, targets)
                total_loss += loss.item()

                # Metrics
                metrics.update(predictions, targets)

        # Average metrics
        epoch_metrics = metrics.get_scores()
        epoch_metrics["loss"] = total_loss / len(self.val_loader)

        return epoch_metrics

    def fit(self, num_epochs: int = NUM_EPOCHS):
        """
        Train model for specified epochs

        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Loss function: DiceBCE (dice={DICE_LOSS_WEIGHT}, bce={BCE_LOSS_WEIGHT})")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log metrics
            logger.info(f"\nEpoch {self.current_epoch} Summary:")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Train Dice (mean): {train_metrics.get('dice_mean', 0.0):.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Val Dice (mean): {val_metrics.get('dice_mean', 0.0):.4f}")

            # TensorBoard logging
            if self.writer:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, self.current_epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, self.current_epoch)

            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get("dice_mean", 0.0))
            elif self.scheduler:
                self.scheduler.step()

            # Save best model
            val_dice = val_metrics.get("dice_mean", 0.0)
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint(self.best_model_path)
                self.patience_counter = 0
                logger.info(f"  ✓ New best model saved (Dice: {val_dice:.4f})")
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if (self.current_epoch % SAVE_CHECKPOINT_EVERY_N_EPOCHS == 0):
                checkpoint_path = (
                    CHECKPOINTS_DIR / f"{self.experiment_name}_epoch{self.current_epoch}.pth"
                )
                self.save_checkpoint(checkpoint_path)

            # Early stopping
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping triggered after {self.current_epoch} epochs"
                )
                break

        logger.info("Training complete!")
        logger.info(f"Best validation Dice: {self.best_val_dice:.4f}")

        # Save final metrics
        self.save_summary()

    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "config": get_config_dict(),
        }

        torch.save(checkpoint, path)

    def save_summary(self):
        """Save training summary"""
        summary = {
            "experiment_name": self.experiment_name,
            "date": datetime.now().isoformat(),
            "total_epochs": self.current_epoch,
            "best_val_dice": self.best_val_dice,
            "best_model_path": str(self.best_model_path),
            "config": get_config_dict(),
        }

        summary_path = OUTPUTS_DIR / f"{self.experiment_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Brain Tumor Segmentation Model"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=EXPERIMENT_NAME,
        help="Experiment name for logging",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/BraTS",
        help="Dataset directory",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of epochs",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        choices=["cuda", "mps", "cpu"],
        help="Device to train on",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating Attention U-Net model...")
    model = AttentionUNet3D(
        in_channels=NUM_INPUT_CHANNELS,
        out_channels=NUM_CLASSES,
        use_attention=USE_ATTENTION_GATES,
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # Create dataloaders
    logger.info(f"Loading dataset from {args.data_dir}...")
    try:
        train_loader, val_loader, test_loader = BraTS2021DataLoader.create_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            augment_train=True,
            augment_val=False,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Run: python scripts/download_data.py --create_mock")
        return

    # Create trainer and train
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        device,
        experiment_name=args.experiment_name,
    )

    trainer.fit(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
