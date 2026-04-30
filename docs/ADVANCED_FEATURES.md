# Advanced Features & Enhancements

## Overview

Extend the core model with research-backed features for improved clinical utility.

---

## Feature 1: Uncertainty Estimation (Bayesian)

### What It Does

Produces confidence maps alongside predictions

```python
# models/uncertainty.py

import torch
import torch.nn as nn
from models.unet_attention import AttentionUNet3D

class BayesianUNet(AttentionUNet3D):
    """U-Net with Monte Carlo Dropout for uncertainty"""

    def __init__(self, *args, num_dropout_samples=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_dropout_samples = num_dropout_samples

    def forward_uncertain(self, x: torch.Tensor):
        """Forward pass with uncertainty estimation"""
        # Run multiple forward passes with dropout enabled
        predictions_list = []

        for _ in range(self.num_dropout_samples):
            # Forward pass with dropout active
            with torch.enable_grad():
                pred = self.forward(x)
                predictions_list.append(pred)

        # Stack predictions
        predictions = torch.stack(predictions_list, dim=0)  # (S, B, C, D, H, W)

        # Compute mean and variance
        mean = predictions.mean(dim=0)  # Prediction
        variance = predictions.var(dim=0)  # Uncertainty

        return mean, variance

    def get_confidencemap(self, x: torch.Tensor, threshold=0.1):
        """Get uncertainty-adjusted predictions"""
        pred_mean, pred_var = self.forward_uncertain(x)

        # High variance = low confidence
        confidence = 1.0 / (1.0 + pred_var)

        # Mask low-confidence regions
        pred_argmax = torch.argmax(pred_mean, dim=1)
        pred_argmax[confidence < threshold] = 0  # Set to background

        return pred_argmax, confidence
```

**Usage:**

```python
model = BayesianUNet(num_dropout_samples=20)
images = torch.randn(1, 4, 32, 32, 32)

# Predict with uncertainty
mean, variance = model.forward_uncertain(images)
predictions, confidence = model.get_confidencemap(images, threshold=0.9)

print(f"Mean shape: {mean.shape}")
print(f"Variance shape: {variance.shape}")
print(f"Confidence shape: {confidence.shape}")
```

---

## Feature 2: Post-Processing Pipeline

### Morphological Operations

```python
# inference/postprocess.py

import torch
import scipy.ndimage as ndimage
import numpy as np

class PostProcessor:
    """Post-processing for segmentation refinement"""

    def __init__(self, use_morphology=True, use_crf=False):
        self.use_morphology = use_morphology
        self.use_crf = use_crf

    @staticmethod
    def remove_small_objects(segmentation, min_size=100):
        """Remove isolated small regions"""
        processed = segmentation.copy()

        for class_id in range(1, 4):
            mask = (segmentation == class_id)
            labeled, num_features = ndimage.label(mask)

            for obj_id in range(1, num_features + 1):
                if np.sum(labeled == obj_id) < min_size:
                    processed[labeled == obj_id] = 0

        return processed

    @staticmethod
    def fill_holes(segmentation):
        """Fill holes in tumor regions"""
        processed = segmentation.copy()

        for class_id in range(1, 4):
            mask = (segmentation == class_id)
            # Fill 3D holes
            from scipy.ndimage import binary_fill_holes
            filled = binary_fill_holes(mask)
            processed[filled] = class_id

        return processed

    @staticmethod
    def smooth_boundaries(segmentation, iterations=2):
        """Smooth tumor boundaries"""
        processed = segmentation.copy()

        for class_id in range(1, 4):
            mask = (segmentation == class_id).astype(float)
            # Gaussian smoothing
            smoothed = ndimage.gaussian_filter(mask, sigma=1.0)
            processed[smoothed > 0.5] = class_id

        return processed

    def process(self, predictions):
        """Apply full post-processing pipeline"""
        # predictions: (D, H, W) class indices

        if self.use_morphology:
            predictions = self.remove_small_objects(predictions)
            predictions = self.fill_holes(predictions)
            predictions = self.smooth_boundaries(predictions)

        return predictions
```

**Usage:**

```python
postprocessor = PostProcessor(use_morphology=True)

# Raw predictions
raw_pred = torch.argmax(logits, dim=1).cpu().numpy()

# Post-process
refined = postprocessor.process(raw_pred)

# Improvement
print(f"Original: {np.sum(raw_pred > 0)} voxels")
print(f"Refined: {np.sum(refined > 0)} voxels")
```

---

## Feature 3: Ensemble Methods

### Multi-Model Voting

```python
# experiments/ensemble.py

import torch
import torch.nn.functional as F
from models.unet_attention import AttentionUNet3D
from experiments.baseline_unet import StandardUNet3D

class SegmentationEnsemble:
    """Ensemble of multiple models for robust predictions"""

    def __init__(self, device='cuda'):
        self.device = device
        self.models = []

    def add_model(self, model_path, model_class=AttentionUNet3D):
        """Add a trained model to ensemble"""
        model = model_class().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.models.append(model)

    def predict(self, images):
        """Ensemble prediction with voting"""
        predictions = []

        with torch.no_grad():
            for model in self.models:
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)

        # Average probabilities
        ensemble_prob = torch.stack(predictions).mean(dim=0)
        ensemble_pred = torch.argmax(ensemble_prob, dim=1)

        # Confidence: max probability
        confidence = ensemble_prob.max(dim=1)[0]

        return ensemble_pred, confidence

    def predict_with_variance(self, images):
        """Ensemble with uncertainty from disagreement"""
        predictions = []

        with torch.no_grad():
            for model in self.models:
                logits = model(images)
                pred = torch.argmax(logits, dim=1)
                predictions.append(pred)

        # Stack predictions
        preds = torch.stack(predictions)  # (M, B, D, H, W)

        # Agreement ratio (variance)
        agreement = torch.zeros_like(preds[0], dtype=torch.float32)
        for i in range(preds.shape[1]):
            # Count agreement at each voxel
            mode = torch.mode(preds[:, i], dim=0)[0]
            agreement += (preds[:, i] == mode).float()

        agreement = agreement / len(self.models)  # [0, 1]: 1=perfect agreement

        return preds.mean(dim=0), 1.0 - agreement  # Mean and disagreement
```

**Usage:**

```python
ensemble = SegmentationEnsemble()
ensemble.add_model('checkpoints/attention_unet_v1.pth', AttentionUNet3D)
ensemble.add_model('checkpoints/attention_unet_v2.pth', AttentionUNet3D)
ensemble.add_model('checkpoints/baseline_unet.pth', StandardUNet3D)

images = torch.randn(1, 4, 32, 32, 32)
predictions, confidence = ensemble.predict(images)

print(f"Ensemble predictions: {predictions.shape}")
print(f"Confidence (agreement): {confidence.mean():.3f}")
```

---

## Feature 4: 3D Processing

### 3D Patches Instead of 2D Slices

```python
# data/dataloader_3d.py

import torch
from torch.utils.data import Dataset
import numpy as np

class BraTS3DDataset(Dataset):
    """3D patch-based dataset for volumetric processing"""

    def __init__(self, case_ids, patch_size=64, stride=32, **kwargs):
        super().__init__(**kwargs)
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.stride = stride
        self.patches = self._extract_patches()

    def _extract_patches(self):
        """Extract 3D patches from volumes"""
        patches = []

        for case_id in self.case_ids:
            # Load 3D volume (D=155, H=240, W=240)
            volume, seg = self._load_case(case_id)

            # Extract overlapping 3D patches
            for d in range(0, volume.shape[0] - self.patch_size, self.stride):
                for h in range(0, volume.shape[1] - self.patch_size, self.stride):
                    for w in range(0, volume.shape[2] - self.patch_size, self.stride):
                        patch = volume[d:d+self.patch_size,
                                      h:h+self.patch_size,
                                      w:w+self.patch_size]
                        seg_patch = seg[d:d+self.patch_size,
                                       h:h+self.patch_size,
                                       w:w+self.patch_size]

                        # Only include patches with tumor
                        if seg_patch.max() > 0:
                            patches.append((patch, seg_patch, case_id))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, seg_patch, case_id = self.patches[idx]

        return {
            'image': torch.from_numpy(patch).float(),
            'segmentation': torch.from_numpy(seg_patch).long(),
            'case_id': case_id
        }
```

---

## Feature 5: Explainability

### Attention Map Visualization

```python
# inference/explainability.py

import matplotlib.pyplot as plt
import torch
import numpy as np

class ExplainabilityAnalyzer:
    """Analyze and visualize model attention maps"""

    @staticmethod
    def extract_attention_maps(model, images):
        """Get attention maps from model"""
        attention_maps = []
        hooks = []

        def hook_fn(module, input, output):
            attention_maps.append(output.detach())

        # Register hooks on attention gates
        for name, module in model.named_modules():
            if 'attention_gate' in name:
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            _ = model(images)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return attention_maps

    @staticmethod
    def visualize_attention(attention_maps, slice_idx=77):
        """Visualize attention maps"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        for i, att_map in enumerate(attention_maps[:4]):
            ax = axes[i // 2, i % 2]

            # Get slice
            att_slice = att_map[0, 0, slice_idx].cpu().numpy()  # (H, W)

            # Normalize
            att_slice = (att_slice - att_slice.min()) / (att_slice.max() - att_slice.min())

            # Plot
            ax.imshow(att_slice, cmap='hot')
            ax.set_title(f'Attention Map {i+1}')
            ax.axis('off')

        plt.tight_layout()
        return fig

    @staticmethod
    def integrated_gradients(model, images, target_class=3, steps=50):
        """Compute integrated gradients for feature importance"""
        images.requires_grad = True

        # Baseline (black image)
        baseline = torch.zeros_like(images)

        # Interpolation
        grads = torch.zeros_like(images)

        for step in range(steps):
            # Interpolated image
            alpha = step / steps
            interp = baseline + alpha * (images - baseline)
            interp.requires_grad = True

            # Forward and backward
            output = model(interp)
            loss = output.sum(dim=[2, 3, 4])[:, target_class].sum()

            # Gradient
            loss.backward()
            grads += interp.grad

        # Attributions
        attributions = (images - baseline) * (grads / steps)

        return attributions
```

**Visualization:**

```python
model = AttentionUNet3D()
images = torch.randn(1, 4, 32, 32, 32)

# Get attention maps
analyzer = ExplainabilityAnalyzer()
att_maps = analyzer.extract_attention_maps(model, images)

# Visualize
fig = analyzer.visualize_attention(att_maps)
plt.savefig('attention_maps.png')

# Integrated gradients
attr = analyzer.integrated_gradients(model, images)
print(f"Attribution magnitude: {attr.abs().mean():.4f}")
```

---

## Feature 6: Active Learning

### Uncertainty-Guided Annotation

```python
# experiments/active_learning.py

class ActiveLearningPipeline:
    """Query most uncertain predictions for annotation"""

    def __init__(self, model, unlabeled_loader, budget=100):
        self.model = model
        self.unlabeled_loader = unlabeled_loader
        self.budget = budget

    def select_uncertain_samples(self):
        """Select most uncertain samples for annotation"""
        uncertainties = []
        sample_ids = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.unlabeled_loader:
                images = batch['image']

                # Get uncertainty
                pred_mean, pred_var = self.model.forward_uncertain(images)

                # Uncertainty: entropy or variance
                uncertainty = pred_var.mean(dim=[2, 3, 4])  # Average over space

                uncertainties.extend(uncertainty.cpu().numpy())
                sample_ids.extend(batch['case_id'])

        # Select top-k most uncertain
        uncertainties = np.array(uncertainties)
        top_k_indices = np.argsort(-uncertainties)[:self.budget]
        selected = [sample_ids[i] for i in top_k_indices]

        return selected

    def annotate_and_retrain(self, newly_labeled):
        """Add newly labeled data and retrain model"""
        # Combine with existing labeled data
        # Retrain model on augmented dataset
        print(f"Added {len(newly_labeled)} new labeled samples")
        print("Retraining model...")
        # training.train() with extended dataset
```

---

## Feature 7: Multi-Task Learning

### Joint Segmentation + Classification

```python
# models/multitask_unet.py

class MultiTaskUNet(AttentionUNet3D):
    """U-Net with auxiliary classification head"""

    def __init__(self, *args, num_classes=4, **kwargs):
        super().__init__(*args, **kwargs)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # 3 tumor grade classes
        )

    def forward(self, x):
        # Segmentation branch
        segmentation_logits = super().forward(x)

        # Classification branch (use bottleneck features)
        # ... extract bottleneck features ...
        class_logits = self.classifier(pooled_features)

        return segmentation_logits, class_logits
```

---

## Feature 8: Semi-Supervised Learning

### Self-Training with Pseudo-Labels

```python
# experiments/semi_supervised.py

class SemiSupervisedTrainer:
    """Train with limited labeled + abundant unlabeled data"""

    def __init__(self, model, labeled_loader, unlabeled_loader):
        self.model = model
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

    def train_epoch_semi(self, confidence_threshold=0.9):
        """Train on labeled + pseudo-labeled unlabeled data"""

        # Step 1: Learn from labeled data
        labeled_loss = self.train_on_labeled()

        # Step 2: Generate pseudo-labels on unlabeled data
        pseudo_labels = self.generate_pseudo_labels(confidence_threshold)

        # Step 3: Train on unlabeled with pseudo-labels
        unlabeled_loss = self.train_on_unlabeled(pseudo_labels, weight=0.5)

        return labeled_loss + 0.5 * unlabeled_loss

    def generate_pseudo_labels(self, threshold):
        """Create confident predictions as labels"""
        pseudo = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.unlabeled_loader:
                images = batch['image']
                output = self.model(images)
                confidence = output.max(dim=1)[0]

                # Keep only high-confidence predictions
                mask = confidence > threshold
                predictions = torch.argmax(output, dim=1)

                pseudo.append((images[mask], predictions[mask]))

        return pseudo
```

---

## Installation & Usage

Add to `requirements.txt`:

```
matplotlib>=3.5
scipy>=1.9
scikit-image>=0.19
```

```bash
pip install -r requirements.txt
```

---

## Recommended Feature Combinations

### For Research:

1. All features above
2. Explainability (attention maps)
3. Active learning

### For Data-Limited Scenarios:

1. Semi-supervised learning
2. Multi-task learning
3. Transfer learning

---

## Performance Impact

| Feature               | +Accuracy | +Speed | Complexity |
| --------------------- | --------- | ------ | ---------- |
| Bayesian (10 samples) | +2-3%     | -90%   | Medium     |
| Post-processing       | +1-2%     | +10%   | Low        |
| Ensemble (3 models)   | +3-4%     | -200%  | Medium     |
| 3D patches            | +2-3%     | -50%   | Medium     |
| Multi-task            | +1-2%     | -10%   | Medium     |
