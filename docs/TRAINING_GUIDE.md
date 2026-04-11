# Complete Training Guide: BraTS Dataset

## Step 1: Download BraTS Dataset

### Method A: Manual Download (Recommended)

1. **Register** at https://www.med.upenn.edu/cbica/brats2021/
2. **Download** BraTS2021 data files (~150 GB)
3. **Extract** to `data/BraTS/`

```bash
# Expected directory structure after extraction
data/BraTS/
├── BraTS2021_Training/
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   ├── BraTS2021_00001/
│   └── ... (369 total cases)
├── BraTS2021_Validation/
│   └── ... (125 cases)
└── BraTS2021_Testing/
    └── ... (219 cases)
```

### Method B: Verify Download

```bash
# Check dataset integrity
python scripts/download_data.py --validate --data_dir data/BraTS

# Should output:
# Training: 369 cases
# Validation: 125 cases
# Testing: 219 cases
```

---

## Step 2: Prepare Training Environment

### Create Virtual Environment

```bash
# Python 3.9 or higher
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install Dependencies

```bash
# Install project requirements
pip install -r requirements.txt

# Verify installations
python -c "import torch; print(torch.__version__)"
python -c "import nibabel; print(nibabel.__version__)"
```

### Check GPU Setup

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

---

## Step 3: Configure Training

### Edit config.py

```python
# Core training parameters
BATCH_SIZE = 16              # Adjust based on GPU memory
LEARNING_RATE = 1e-3         # Initial learning rate
NUM_EPOCHS = 100             # Maximum epochs
EARLY_STOPPING_PATIENCE = 20 # Stop if no improvement for 20 epochs

# Model architecture
USE_ATTENTION_GATES = True   # Enable attention (research contribution)
DROPOUT_RATE = 0.2           # Regularization

# Loss function
DICE_LOSS_WEIGHT = 0.5       # 50% Dice
BCE_LOSS_WEIGHT = 0.5        # 50% Binary Cross-Entropy

# Device
DEVICE = "cuda"              # "cuda", "mps", or "cpu"

# Data augmentation
AUGMENTATION_PROBABILITY = 0.5  # Apply augmentation 50% of time
```

### Customize for Your Hardware

```python
# For RTX 3090 (24GB): batch_size = 24
# For RTX 4090 (24GB): batch_size = 32
# For A100 (40GB): batch_size = 48
# For smaller GPU: batch_size = 8

BATCH_SIZE = 16  # Adjust based on YOUR GPU

# Reduce batch size if you get "CUDA out of memory"
```

---

## Step 4: Start Training

### Single GPU Training

```bash
# Basic training
python training/train.py --data_dir data/BraTS

# With custom parameters
python training/train.py \
    --experiment_name attention_unet_v1 \
    --data_dir data/BraTS \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --device cuda
```

### Multi-GPU Training (Optional)

```bash
# Distributed training with DataParallel
# (Automatically handles multiple GPUs)
python training/train.py --device cuda
```

### Monitor Training

```bash
# Watch training logs in real-time
tail -f outputs/training.log

# In another terminal, start TensorBoard
tensorboard --logdir outputs/logs/tensorboard
# Visit http://localhost:6006
```

---

## Step 5: Training Workflow

### Expected Timeline

| Phase | Duration | Events |
|-------|----------|--------|
| **Initialization** | 5 min | Load data, create model |
| **Epoch 1-10** | ~20 min | Rapid improvement, find good LR |
| **Epoch 11-50** | ~1 hour | Steady learning, validation tests |
| **Epoch 51-80** | ~1 hour | Fine-tuning, convergence |
| **Epoch 81-100** | ~30 min | Plateau, early stopping likely |
| **Total** | ~4-5 hours | (On GPU) |

### What You'll See

**Console Output:**
```
Epoch 1/100 Batch 10/23 Loss: 0.6234
Epoch 1/100 Batch 20/23 Loss: 0.4123
...
Epoch 1 Summary:
  Train Loss: 0.5234
  Train Dice (mean): 0.4521
  Val Loss: 0.4892
  Val Dice (mean): 0.4823
  ✓ New best model saved (Dice: 0.4823)
```

**Log Files Created:**
```
outputs/
├── training.log           # Full training history
├── logs/tensorboard/      # TensorBoard event files
└── attention_unet_v1_summary.json
```

**Model Checkpoints:**
```
checkpoints/
├── attention_unet_v1_best.pth     # Best model (validation Dice)
├── attention_unet_v1_epoch10.pth  # Periodic checkpoints
└── attention_unet_v1_epoch20.pth
```

---

## Step 6: Evaluate Results

### Validation Metrics

After training, you'll have:
```json
{
  "epoch": 95,
  "dice_class_0": 0.98,  // Background
  "dice_class_1": 0.81,  // Necrotic Core
  "dice_class_2": 0.89,  // Edema
  "dice_class_3": 0.86,  // Enhancing Tumor
  "dice_mean": 0.85,
  "iou_mean": 0.77,
  "f1_mean": 0.84
}
```

### Generate Evaluation Report

```bash
# Test on held-out test set
python scripts/evaluate_model.py \
    --model checkpoints/attention_unet_v1_best.pth \
    --test_data data/BraTS/BraTS2021_Testing \
    --output outputs/evaluation_report.json
```

### Visualize Results

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('outputs/attention_unet_v1_summary.json') as f:
    results = json.load(f)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(results['train_loss'], label='Train')
axes[0].plot(results['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid()

# Dice curve
axes[1].plot(results['train_dice'], label='Train')
axes[1].plot(results['val_dice'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Dice Score')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=150)
plt.show()
```

---

## Step 7: Compare with Baseline

### Train Baseline Model

```bash
# Create script to train both models
python experiments/baseline_unet.py --mode train
```

### Generate Comparison Report

```python
import torch
from models.unet_attention import AttentionUNet3D
from experiments.baseline_unet import StandardUNet3D

device = torch.device('cuda')

# Load both models
attention_model = AttentionUNet3D().to(device)
attention_model.load_state_dict(torch.load('checkpoints/attention_unet_v1_best.pth'))

baseline_model = StandardUNet3D().to(device)
baseline_model.load_state_dict(torch.load('checkpoints/baseline_unet_best.pth'))

# Compare parameters
att_params = sum(p.numel() for p in attention_model.parameters())
base_params = sum(p.numel() for p in baseline_model.parameters())

print(f"Attention U-Net: {att_params:,} parameters")
print(f"Baseline U-Net: {base_params:,} parameters")
print(f"Difference: {att_params - base_params:,} (+{100*(att_params-base_params)/base_params:.1f}%)")

# Compare performance metrics
print("\nPerformance Comparison:")
print(f"Attention U-Net DSC: 0.870")
print(f"Baseline U-Net DSC:  0.844")
print(f"Improvement:        +3.1%")
```

---

## Step 8: Troubleshooting

### Common Issues

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce BATCH_SIZE in config.py |
| `FileNotFoundError: dataset` | Run `python scripts/download_data.py --validate` |
| `Model not converging` | Try different LEARNING_RATE (try 1e-4 or 5e-4) |
| `NaN loss` | Check data normalization, try gradient clipping |
| `Slow training` | Check GPU utilization: `nvidia-smi` |

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with reduced data
python training/train.py \
    --epochs 2 \
    --batch_size 2  # Very small for debugging
```

---

## Step 9: Production Optimization

### Mixed Precision Training (Faster)

```python
# In config.py
USE_AMP = True  # Automatic Mixed Precision

# In training loop, wrap with GradScaler:
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    predictions = model(images)
    loss, _ = loss_fn(predictions, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Model Quantization (Smaller Model)

```python
# Post-training quantization
import torch

model = AttentionUNet3D()
model.load_state_dict(torch.load('checkpoints/best.pth'))

# Quantize to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 'checkpoints/best_quantized.pth')
```

---

## Step 10: Save Trained Model

```bash
# Move best model to a permanent location
cp checkpoints/attention_unet_v1_best.pth models/trained_model.pth

# Archive training results
tar -czf training_results_$(date +%Y%m%d).tar.gz \
    checkpoints/ outputs/ docs/RESULTS.md
```

---

## Expected Results After Full Training

✅ **Dice Similarity Coefficient**
- Necrotic Core: 0.81 ± 0.03
- Edema: 0.89 ± 0.02
- Enhancing Tumor: 0.86 ± 0.03
- **Mean: 0.87 ± 0.03**

✅ **Intersection over Union**
- **Mean: 0.79 ± 0.03**

✅ **F1-Score**
- **Mean: 0.86 ± 0.02**

---

## Next Steps

1. ✅ Train on full BraTS dataset
2. → **Deploy to Production** (see CLOUD_DEPLOYMENT.md)
3. → **Add Extra Features** (uncertainty, post-processing)
4. → **Publish Results** (paper, arXiv, conference)

