"""
Evaluation Metrics for Brain Tumor Segmentation
Dice Similarity Coefficient, IoU, F1-Score, and more
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Compute segmentation evaluation metrics"""

    @staticmethod
    def dice_score(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-7,
        class_id: Optional[int] = None,
    ) -> float:
        """
        Dice Similarity Coefficient (DSC)
        Formula: DSC = 2|X∩Y| / (|X| + |Y|)

        Args:
            predictions: (B, H, W, D) or (B, C, H, W, D) - class indices or one-hot
            targets: (B, H, W, D) - ground truth indices
            smooth: Smoothing constant
            class_id: Specific class to compute (None = all classes)

        Returns:
            DSC score (0-1, higher is better)
        """
        if predictions.dim() == 5:  # (B, C, D, H, W)
            predictions = torch.argmax(predictions, dim=1)

        if class_id is not None:
            pred_mask = (predictions == class_id).float()
            target_mask = (targets == class_id).float()
        else:
            pred_mask = (predictions > 0).float()
            target_mask = (targets > 0).float()

        intersection = (pred_mask * target_mask).sum().float()
        union = pred_mask.sum() + target_mask.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()

    @staticmethod
    def iou_score(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_id: Optional[int] = None,
    ) -> float:
        """
        Intersection over Union (Jaccard Index)
        Formula: IoU = |X∩Y| / |X∪Y|

        Args:
            predictions: Predicted segmentation
            targets: Ground truth segmentation
            class_id: Specific class (None = all)

        Returns:
            IoU score (0-1)
        """
        if predictions.dim() == 5:
            predictions = torch.argmax(predictions, dim=1)

        if class_id is not None:
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
        else:
            pred_mask = predictions > 0
            target_mask = targets > 0

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        iou = intersection / (union + 1e-7)
        return iou.item()

    @staticmethod
    def f1_score(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_id: Optional[int] = None,
    ) -> float:
        """
        F1 Score
        Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Args:
            predictions: Predicted segmentation
            targets: Ground truth segmentation
            class_id: Specific class

        Returns:
            F1 score (0-1)
        """
        if predictions.dim() == 5:
            predictions = torch.argmax(predictions, dim=1)

        if class_id is not None:
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
        else:
            pred_mask = predictions > 0
            target_mask = targets > 0

        tp = (pred_mask & target_mask).sum().float()
        fp = (pred_mask & ~target_mask).sum().float()
        fn = (~pred_mask & target_mask).sum().float()

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1.item()

    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int = 4,
    ) -> Dict[str, float]:
        """
        Compute all metrics for all classes

        Args:
            predictions: (B, C, D, H, W) or (B, D, H, W)
            targets: (B, D, H, W)
            num_classes: Number of classes

        Returns:
            Dictionary with per-class and mean metrics
        """
        metrics = {}

        # Ensure predictions are class indices
        if predictions.dim() == 5:
            predictions = torch.argmax(predictions, dim=1)

        # Per-class metrics
        for c in range(num_classes):
            dice = SegmentationMetrics.dice_score(predictions, targets, class_id=c)
            iou = SegmentationMetrics.iou_score(predictions, targets, class_id=c)
            f1 = SegmentationMetrics.f1_score(predictions, targets, class_id=c)

            metrics[f"dice_class_{c}"] = dice
            metrics[f"iou_class_{c}"] = iou
            metrics[f"f1_class_{c}"] = f1

        # Mean metrics (exclude background)
        dice_scores = [metrics[f"dice_class_{c}"] for c in range(1, num_classes)]
        iou_scores = [metrics[f"iou_class_{c}"] for c in range(1, num_classes)]
        f1_scores = [metrics[f"f1_class_{c}"] for c in range(1, num_classes)]

        metrics["dice_mean"] = np.mean(dice_scores) if dice_scores else 0.0
        metrics["iou_mean"] = np.mean(iou_scores) if iou_scores else 0.0
        metrics["f1_mean"] = np.mean(f1_scores) if f1_scores else 0.0

        return metrics

    @staticmethod
    def hausdorff_distance(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_id: int = 0,
        percentile: int = 95,
    ) -> float:
        """
        Hausdorff Distance (95th percentile)
        Measures maximum distance between surfaces

        Args:
            predictions: Predicted segmentation
            targets: Ground truth segmentation
            class_id: Class to compute distance for
            percentile: Percentile for robust Hausdorff

        Returns:
            Hausdorff distance
        """
        if predictions.dim() == 5:
            predictions = torch.argmax(predictions, dim=1)

        pred_mask = (predictions == class_id).float()
        target_mask = (targets == class_id).float()

        # Extract surface points (simplified)
        # In practice, use scipy.ndimage.binary_gradient
        pred_surface = pred_mask.sum() > 0
        target_surface = target_mask.sum() > 0

        if not pred_surface or not target_surface:
            return 0.0

        # Simple implementation (returns max distance)
        # Full implementation would compute true Hausdorff distance
        return float(torch.norm(pred_mask - target_mask).item())


class MetricAggregator:
    """Aggregate metrics across batches"""

    def __init__(self, num_classes: int = 4):
        """
        Args:
            num_classes: Number of segmentation classes
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.dice_scores = {f"class_{c}": [] for c in range(self.num_classes)}
        self.iou_scores = {f"class_{c}": [] for c in range(self.num_classes)}
        self.f1_scores = {f"class_{c}": [] for c in range(self.num_classes)}
        self.batch_count = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update metrics with new batch

        Args:
            predictions: (B, C, D, H, W) or (B, D, H, W)
            targets: (B, D, H, W)
        """
        metrics = SegmentationMetrics.compute_metrics(
            predictions,
            targets,
            self.num_classes,
        )

        for c in range(self.num_classes):
            self.dice_scores[f"class_{c}"].append(metrics[f"dice_class_{c}"])
            self.iou_scores[f"class_{c}"].append(metrics[f"iou_class_{c}"])
            self.f1_scores[f"class_{c}"].append(metrics[f"f1_class_{c}"])

        self.batch_count += 1

    def get_scores(self) -> Dict[str, float]:
        """
        Get aggregated metrics

        Returns:
            Dictionary with mean metrics
        """
        result = {}

        for c in range(self.num_classes):
            dice_list = self.dice_scores[f"class_{c}"]
            iou_list = self.iou_scores[f"class_{c}"]
            f1_list = self.f1_scores[f"class_{c}"]

            result[f"dice_class_{c}"] = np.mean(dice_list) if dice_list else 0.0
            result[f"iou_class_{c}"] = np.mean(iou_list) if iou_list else 0.0
            result[f"f1_class_{c}"] = np.mean(f1_list) if f1_list else 0.0

        # Mean excluding background
        dice_means = [
            result[f"dice_class_{c}"] for c in range(1, self.num_classes)
        ]
        iou_means = [result[f"iou_class_{c}"] for c in range(1, self.num_classes)]
        f1_means = [result[f"f1_class_{c}"] for c in range(1, self.num_classes)]

        result["dice_mean"] = np.mean(dice_means)
        result["iou_mean"] = np.mean(iou_means)
        result["f1_mean"] = np.mean(f1_means)

        return result

    def __repr__(self) -> str:
        """String representation with all metrics"""
        scores = self.get_scores()
        lines = [f"Metrics (batches={self.batch_count}):"]
        for key, value in sorted(scores.items()):
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    batch_size, depth, height, width = 2, 32, 32, 32
    num_classes = 4

    # Create dummy predictions and targets
    predictions = torch.randint(0, num_classes, (batch_size, depth, height, width))
    targets = torch.randint(0, num_classes, (batch_size, depth, height, width))

    # Test individual metrics
    print("Testing individual metrics...")
    dice = SegmentationMetrics.dice_score(predictions, targets)
    iou = SegmentationMetrics.iou_score(predictions, targets)
    f1 = SegmentationMetrics.f1_score(predictions, targets)

    print(f"  Dice: {dice:.4f}")
    print(f"  IoU: {iou:.4f}")
    print(f"  F1: {f1:.4f}")

    # Test metric aggregator
    print("\nTesting MetricAggregator...")
    aggregator = MetricAggregator(num_classes)

    for _ in range(3):
        pred = torch.randint(0, num_classes, (batch_size, depth, height, width))
        target = torch.randint(0, num_classes, (batch_size, depth, height, width))
        aggregator.update(pred, target)

    print(aggregator)
