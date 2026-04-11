"""
Loss Functions for Brain Tumor Segmentation
Composite Dice-BCE loss with class weighting for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

from config import DICE_LOSS_WEIGHT, BCE_LOSS_WEIGHT, NUM_CLASSES

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Dice Similarity Coefficient Loss

    Benefits:
    - Handles class imbalance better than cross-entropy
    - Directly optimizes the metric we care about (Dice)
    - Works well with small objects (tumors)

    Formula: L = 1 - DSC, where DSC = 2|X∩Y| / (|X| + |Y|)
    """

    def __init__(
        self,
        smooth: float = 1e-7,
        reduction: str = "mean",
        per_class: bool = True,
    ):
        """
        Args:
            smooth: Small constant to avoid division by zero
            reduction: 'mean' or 'sum'
            per_class: Whether to compute loss per class
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.per_class = per_class

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Dice loss

        Args:
            predictions: Model output logits (B, C, D, H, W)
            targets: Target segmentation (B, D, H, W) with class indices
            weights: Optional class weights (C,)

        Returns:
            Scalar loss value
        """
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)

        # Convert targets to one-hot
        batch_size, num_classes, *spatial_dims = predictions.shape
        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=num_classes,
        ).permute(0, *range(2, len(targets.shape) + 1), 1).float()

        # Compute Dice per class
        dice_list = []

        for c in range(num_classes):
            pred_c = predictions[:, c, ...].flatten()
            target_c = targets_one_hot[..., c].flatten()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

            if weights is not None:
                dice = dice * weights[c]

            dice_list.append(dice)

        # Aggregate
        loss = 1.0 - torch.stack(dice_list).mean()

        if self.per_class:
            logger.debug(f"Per-class Dice: {[f'{d:.4f}' for d in dice_list]}")

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Benefits:
    - Down-weights easy examples
    - Focuses on hard examples
    - Useful for extreme class imbalance

    Formula: FL = -α(1-p_t)^γ * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Weighting factor for foreground class
            gamma: Exponent for focusing parameter
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal loss

        Args:
            predictions: Model output logits (B, C, D, H, W)
            targets: Target segmentation (B, D, H, W)

        Returns:
            Scalar loss value
        """
        # Apply softmax
        p = F.softmax(predictions, dim=1)

        # Get class probabilities
        p_t = p.gather(1, targets.unsqueeze(1))

        # Compute focal loss
        focal_loss = -(1 - p_t) ** self.gamma * torch.log(p_t + 1e-7)

        # Apply alpha weighting
        focal_loss = self.alpha * focal_loss + (1 - self.alpha) * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DiceBCELoss(nn.Module):
    """
    Composite Dice + Binary Cross-Entropy Loss

    Combines two complementary losses:
    - Dice: Handles class imbalance, optimizes intersection/union
    - BCE: Provides pixel-level classification supervision

    This combination is particularly effective for medical image segmentation
    with extreme class imbalance.
    """

    def __init__(
        self,
        dice_weight: float = DICE_LOSS_WEIGHT,
        bce_weight: float = BCE_LOSS_WEIGHT,
        smooth: float = 1e-7,
        reduction: str = "mean",
    ):
        """
        Args:
            dice_weight: Weight for Dice loss (typically 0.5)
            bce_weight: Weight for BCE loss (typically 0.5)
            smooth: Smoothing constant for Dice
            reduction: 'mean' or 'sum'
        """
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice = DiceLoss(smooth=smooth, reduction=reduction)
        self.bce = nn.CrossEntropyLoss(reduction=reduction)

        logger.info(
            f"Initialized DiceBCELoss with "
            f"dice_weight={dice_weight}, bce_weight={bce_weight}"
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute composite loss and component losses

        Args:
            predictions: Model output logits (B, C, D, H, W)
            targets: Target segmentation (B, D, H, W)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Compute individual losses
        dice_loss = self.dice(predictions, targets)
        bce_loss = self.bce(predictions, targets)

        # Composite loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        components = {
            "dice": dice_loss.item(),
            "bce": bce_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, components


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss with per-class weighting

    Useful for balancing class importance when:
    - Some classes are more important than others
    - Background class dominates
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            class_weights: Tensor of shape (C,) with class weights
            reduction: 'mean' or 'sum'
        """
        super().__init__()

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction=reduction,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, D, H, W)
            targets: (B, D, H, W)

        Returns:
            Weighted cross-entropy loss
        """
        return self.ce_loss(predictions, targets)


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss

    Directly optimizes IoU metric
    Particularly effective for imbalanced segmentation tasks

    Reference: "The Lovász-Softmax loss: A tractable surrogate for the
               optimization of the intersection-over-union measure in
               neural networks" (Berman et al., 2018)
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, D, H, W)
            targets: (B, D, H, W)

        Returns:
            Lovász-Softmax loss
        """
        # Apply softmax
        probs = F.softmax(predictions, dim=1)

        return self.lovasz_softmax(probs, targets)

    @staticmethod
    def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None):
        """
        Multi-class Lovász-softmax loss
        probas: (B, C, H, W, ...) Variable, class probabilities
        labels: (B, H, W, ...) Tensor, ground truth labels (0 <= labels[i] <= C-1)
        """
        if per_image:
            loss = mean(
                lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0)), ignore=ignore)
                for prob, lab in zip(probas, labels)
            )
        else:
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels), only_present=only_present, ignore=ignore)
        return loss

    @staticmethod
    def flatten_probas(probas, labels, ignore=None):
        if probas.dim() == 3:
            with_depth = False
            probas = probas.unsqueeze(2)
        else:
            with_depth = True

        B, C, *rest = probas.shape
        probas = probas.permute(0, *range(2, len(probas.shape)), 1).contiguous()
        probas = probas.view(-1, C)

        labels = labels.view(-1)

        if ignore is None:
            return probas, labels

        valid = labels != ignore
        vprobas = probas[valid.nonzero(as_tuple=True)]
        vlabels = labels[valid]
        return vprobas, vlabels


def compute_class_weights(
    dataset_loader,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """
    Compute per-class weights based on class frequency in dataset

    Args:
        dataset_loader: DataLoader providing batches
        num_classes: Number of classes

    Returns:
        Tensor of shape (num_classes,) with inverse frequency weights
    """
    class_counts = torch.zeros(num_classes)

    for batch in dataset_loader:
        targets = batch["segmentation"]
        for c in range(num_classes):
            class_counts[c] += (targets == c).sum().item()

    # Inverse frequency weighting
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes

    logger.info(f"Computed class weights: {class_weights}")

    return class_weights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test losses
    batch_size, num_classes, depth, height, width = 2, 4, 32, 32, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = torch.randn(batch_size, num_classes, depth, height, width).to(device)
    targets = torch.randint(0, num_classes, (batch_size, depth, height, width)).to(device)

    # Test Dice Loss
    print("Testing Dice Loss...")
    dice = DiceLoss().to(device)
    dice_loss = dice(predictions, targets)
    print(f"  Dice Loss: {dice_loss:.4f}")

    # Test BCE Loss
    print("\nTesting Cross-Entropy Loss...")
    ce = nn.CrossEntropyLoss().to(device)
    ce_loss = ce(predictions, targets)
    print(f"  CE Loss: {ce_loss:.4f}")

    # Test Composite Loss
    print("\nTesting Composite DiceBCE Loss...")
    composite = DiceBCELoss().to(device)
    total_loss, components = composite(predictions, targets)
    print(f"  Dice: {components['dice']:.4f}")
    print(f"  BCE: {components['bce']:.4f}")
    print(f"  Total: {components['total']:.4f}")

    print("\n✓ All loss functions working!")
