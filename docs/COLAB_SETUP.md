# Google Colab Setup & Training Guide

Complete guide to run Brain Tumor Segmentation in Google Colab (free GPU access).

---

## Quick Start (Copy-Paste Ready)

### Step 1: Create Colab Notebook

Go to: https://colab.research.google.com/

---

## Step 2: Enable GPU

**Cell 1:** Enable GPU
```python
# Check GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**To enable GPU:**
- Menu: Runtime → Change runtime type → GPU (T4/P100/V100/A100 if available)
- Click Save

---

## Step 3: Mount Google Drive

**Cell 2:** Mount Drive for persistent storage
```python
from google.colab import drive
drive.mount('/content/drive')

# Create working directory
import os
os.makedirs('/content/drive/MyDrive/brain-tumor-seg', exist_ok=True)
os.chdir('/content/drive/MyDrive/brain-tumor-seg')
print("Working directory:", os.getcwd())
```

---

## Step 4: Clone Repository

**Cell 3:** Clone and setup project
```python
import subprocess
import sys

# Clone repository
!git clone https://github.com/YOUR_USERNAME/Brain-Tumour-Segmentation.git
%cd Brain-Tumour-Segmentation

# Install dependencies
!pip install -r requirements.txt -q

print("✓ Repository cloned and dependencies installed")
```

**Replace `YOUR_USERNAME` with your GitHub username**

---

## Step 5: Setup BraTS Dataset

### Option A: Use Mock Dataset (Recommended for Testing)

**Cell 4:** Create mock dataset
```python
# Create mock BraTS dataset for testing
!python scripts/download_data.py --create_mock --num_cases 20 --output_dir data/BraTS_MOCK

print("✓ Mock dataset created with 20 cases")
```

**This creates synthetic data (~500MB) for quick testing**

### Option B: Download Real BraTS Dataset

**Cell 5:** Download real dataset
```python
# Step A: Manually download from https://www.med.upenn.edu/cbica/brats2021/
# Save .zip file to Google Drive

# Step B: Extract in Colab
!unzip '/content/drive/MyDrive/brain-tumor-seg/BraTS2021_Training_Data.zip' -d data/BraTS

print("✓ BraTS dataset extracted")
```

**For large datasets (~150 GB):**
- Download to Drive first (takes 1-2 hours on fast internet)
- Extract in Colab using the command above

---

## Step 6: Configure Training

**Cell 6:** Adjust config for Colab
```python
# Edit config.py for Colab GPU
config_changes = """
# In config.py, update:

BATCH_SIZE = 8              # Smaller for free GPU tier
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50             # Reduced for testing
EARLY_STOPPING_PATIENCE = 10

DEVICE = "cuda"             # Uses available GPU
PIN_MEMORY = True
NUM_WORKERS = 2
"""

print("Update config.py with above settings")
```

Or programmatically:
```python
# Read config.py
with open('config.py', 'r') as f:
    config = f.read()

# Modify parameters
config = config.replace('BATCH_SIZE = 16', 'BATCH_SIZE = 8')
config = config.replace('NUM_EPOCHS = 100', 'NUM_EPOCHS = 50')
config = config.replace('NUM_WORKERS = 4', 'NUM_WORKERS = 2')

# Write back
with open('config.py', 'w') as f:
    f.write(config)

print("✓ Config updated for Colab")
```

---

## Step 7: Start Training

**Cell 7:** Run training
```python
import subprocess

# Start training
result = subprocess.run([
    'python', 'training/train.py',
    '--experiment_name', 'colab_attention_unet',
    '--epochs', '50',
    '--batch_size', '8',
    '--device', 'cuda'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

Or simpler approach:
```python
%cd /content/drive/MyDrive/brain-tumor-seg/Brain-Tumour-Segmentation

!python training/train.py \
    --experiment_name colab_v1 \
    --epochs 50 \
    --batch_size 8 \
    --device cuda
```

---

## Step 8: Monitor Training with TensorBoard

**Cell 8:** View training progress
```python
%load_ext tensorboard
%tensorboard --logdir outputs/logs/tensorboard
```

This shows real-time:
- Loss curves
- Validation metrics
- Training progress

---

## Step 9: Save Checkpoints to Drive

**Cell 9:** Auto-save important files
```python
import shutil
import os

# Copy best model to Drive
source = 'checkpoints/colab_attention_unet_best.pth'
dest = '/content/drive/MyDrive/brain-tumor-seg/colab_attention_unet_best.pth'

if os.path.exists(source):
    shutil.copy(source, dest)
    print(f"✓ Model saved to Drive: {dest}")
else:
    print("Model file not found yet, still training...")

# Also save training summary
source = 'outputs/colab_v1_summary.json'
dest = '/content/drive/MyDrive/brain-tumor-seg/colab_v1_summary.json'

if os.path.exists(source):
    shutil.copy(source, dest)
    print(f"✓ Summary saved")
```

---

## Step 10: Run Inference

**Cell 10:** Test prediction
```python
import torch
from models.unet_attention import AttentionUNet3D

# Load model
device = torch.device('cuda')
model = AttentionUNet3D().to(device)
model.load_state_dict(torch.load('checkpoints/colab_attention_unet_best.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 4, 32, 32, 32).to(device)

# Run inference
with torch.no_grad():
    output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print("✓ Inference test passed")
```

---

## Complete Full Training Script

**Cell (Master):** Combined training notebook
```python
# ============================================================================
# BRAIN TUMOR SEGMENTATION - GOOGLE COLAB COMPLETE TRAINING SCRIPT
# ============================================================================

# 1. SETUP GPU & STORAGE
print("=" * 60)
print("STEP 1: Setting up GPU & Storage")
print("=" * 60)

import torch
import os
from google.colab import drive

drive.mount('/content/drive', force_remount=True)
os.makedirs('/content/drive/MyDrive/brain-tumor-seg', exist_ok=True)
os.chdir('/content/drive/MyDrive/brain-tumor-seg')

print(f"✓ GPU Available: {torch.cuda.is_available()}")
print(f"✓ Working Dir: {os.getcwd()}")

# 2. CLONE & INSTALL
print("\n" + "=" * 60)
print("STEP 2: Clone Repository & Install Dependencies")
print("=" * 60)

if not os.path.exists('Brain-Tumour-Segmentation'):
    !git clone https://github.com/YOUR_USERNAME/Brain-Tumour-Segmentation.git
    
%cd Brain-Tumour-Segmentation
!pip install -r requirements.txt -q
print("✓ Dependencies installed")

# 3. SETUP DATA
print("\n" + "=" * 60)
print("STEP 3: Setup Dataset")
print("=" * 60)

# Option: Create mock data for testing
!python scripts/download_data.py --create_mock --num_cases 20 --output_dir data/BraTS_MOCK
print("✓ Mock dataset created")

# 4. CONFIGURE
print("\n" + "=" * 60)
print("STEP 4: Configure Training")
print("=" * 60)

with open('config.py', 'r') as f:
    config = f.read()

# Optimize for Colab free tier
config = config.replace('BATCH_SIZE = 16', 'BATCH_SIZE = 8')
config = config.replace('NUM_EPOCHS = 100', 'NUM_EPOCHS = 50')
config = config.replace('NUM_WORKERS = 4', 'NUM_WORKERS = 2')
config = config.replace('PIN_MEMORY = True', 'PIN_MEMORY = True')

with open('config.py', 'w') as f:
    f.write(config)

print("✓ Config optimized for Colab")

# 5. TRAIN
print("\n" + "=" * 60)
print("STEP 5: Start Training")
print("=" * 60)

!python training/train.py \
    --experiment_name colab_attention_unet \
    --epochs 50 \
    --batch_size 8 \
    --device cuda

# 6. SAVE RESULTS
print("\n" + "=" * 60)
print("STEP 6: Save Results to Drive")
print("=" * 60)

import shutil

# Copy model
if os.path.exists('checkpoints/colab_attention_unet_best.pth'):
    shutil.copy(
        'checkpoints/colab_attention_unet_best.pth',
        '/content/drive/MyDrive/brain-tumor-seg/colab_attention_unet_best.pth'
    )
    print("✓ Model saved to Drive")

# Copy summary
if os.path.exists('outputs/colab_attention_unet_summary.json'):
    shutil.copy(
        'outputs/colab_attention_unet_summary.json',
        '/content/drive/MyDrive/brain-tumor-seg/colab_summary.json'
    )
    print("✓ Summary saved to Drive")

print("\n✅ TRAINING COMPLETE!")
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **GPU Timeout (12 hours)** | Upload checkpoint to Drive before timeout. Resume from checkpoint with `--resume` flag |
| **Out of Memory** | Reduce `BATCH_SIZE` to 4 in config.py |
| **Slow Training** | Mock dataset has reduced samples. Use smaller `NUM_EPOCHS` for testing |
| **Drive Not Mounted** | Runtime → Disconnect & reconnect → Remount Drive |
| **Can't Download Dataset** | Use `--create_mock` flag to generate synthetic data instead |
| **Model Not Saving** | Check Drive storage quota (free tier: 15 GB) |

---

## Dataset Size Reference

| Dataset | Size | Time to Download |
|---------|------|----------|
| Mock (20 cases) | ~500 MB | Instant (generated) |
| Mock (100 cases) | ~2.5 GB | 5 min |
| Real BraTS (full) | ~150 GB | 2-4 hours |
| Real BraTS (100 cases) | ~50 GB | 45 min |

**Recommendation:** Start with mock dataset to test pipeline, then download real data.

---

## Advanced: Resume Training from Checkpoint

If training session times out (12 hours):

**Cell:** Resume training
```python
import torch
from training.train import Trainer

# Load checkpoint
checkpoint = torch.load('checkpoints/colab_attention_unet_latest.pth')

# Resume training
trainer = Trainer()
trainer.resume_from_checkpoint(checkpoint)
trainer.fit(num_epochs=100, start_epoch=checkpoint['epoch'])

print("✓ Training resumed from checkpoint")
```

---

## GPU Selection

**Colab Free Tier Speeds:**
- **T4 GPU**: ~1 epoch = 3-4 min (50 epochs = 2-3 hours)
- **P100 GPU**: ~1 epoch = 2 min (50 epochs = 1.5-2 hours)
- **V100 GPU**: ~1 epoch = 1.5 min (50 epochs = 1-1.5 hours)
- **A100 GPU**: ~1 epoch = 1 min (50 epochs = 50 min)

**Note:** A100 requires Colab Pro ($10/month)

---

## Next Steps

1. ✅ Run mock dataset training first (10-15 min)
2. → Download real BraTS dataset (optional)
3. → Train on full dataset (if storage allows)
4. → Download model to local machine
5. → Run local inference or deploy

---

## Save & Share Notebook

After completing:

```python
# Save notebook
from google.colab import files
files.download('Brain-Tumour-Segmentation.ipynb')

# Or share the Colab link directly
print("Notebook saved! Share via: File → Share")
```

---

## Questions?

- GPU too slow? Check runtime GPU selection
- Out of memory? Reduce BATCH_SIZE further
- Training interrupted? Implement checkpoint resuming
- Data too large? Use mock dataset or BraTS subset

**Working in Colab? You're all set to train! 🚀**
