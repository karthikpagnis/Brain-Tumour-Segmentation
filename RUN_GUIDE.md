# Brain Tumor Segmentation - Complete Run Guide

Train and deploy the neural network project from start to finish.

---

## Prerequisites

You need:

1. Google account (for Google Colab - FREE)
2. GitHub account (optional, but recommended)
3. Internet connection
4. No local GPU required (uses Colab's free T4 GPU)

---

## Step 1: Prepare Repository on GitHub

### Option A: Use Existing Repository

If your code is already on GitHub, just use your username.

### Option B: Create New Repository

1. Go to: https://github.com/new
2. Repository name: `Brain-Tumour-Segmentation`
3. Make it PUBLIC (so Colab can access files)
4. Clone to your computer and add all project files

---

## Step 2: Open Google Colab

1. Go to: https://colab.research.google.com/
2. Click: **File → New notebook**
3. Rename: Click on "Untitled" at top → type "Brain Tumor Training"

---

## Step 3: Enable GPU

1. Click: **Runtime** (top menu)
2. Click: **Change runtime type**
3. Select: **GPU** (T4 - free tier)
4. Click: **Save**

---

## Step 4: Run the Cells (Copy-Paste Each One)

Run these 8 cells in order. For each cell:

1. Click in the empty cell area
2. Copy the code below
3. Paste it
4. Press `Shift + Enter` to run

---

## CELL 1: Initialize and Mount Drive

```python
import torch
from google.colab import drive
import os

print("="*60)
print("CELL 1: Initialize and Mount Drive")
print("="*60)

# Check GPU
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")

# Mount Google Drive
print("\nMounting Google Drive...")
drive.mount('/content/drive')

# Create workspace directory
workspace = '/content/drive/MyDrive/brain-tumor-training'
os.makedirs(workspace, exist_ok=True)
os.chdir(workspace)

print(f"\nWorking directory: {os.getcwd()}")
print("Ready to proceed!")
```

---

## CELL 2: Create Project Structure

```python
import os

print("="*60)
print("CELL 2: Create Project Structure")
print("="*60)

# Create all necessary directories
directories = [
    'Brain-Tumour-Segmentation/scripts',
    'Brain-Tumour-Segmentation/training',
    'Brain-Tumour-Segmentation/models',
    'Brain-Tumour-Segmentation/data',
    'Brain-Tumour-Segmentation/outputs',
    'Brain-Tumour-Segmentation/checkpoints',
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created: {directory}")

os.chdir('Brain-Tumour-Segmentation')

print(f"\nWorking directory: {os.getcwd()}")
print("Directory structure created!")
```

---

## CELL 3: Download Project Files

```python
import subprocess
import os

print("="*60)
print("CELL 3: Download Project Files")
print("="*60)

# IMPORTANT: Replace YOUR_USERNAME with your GitHub username
GITHUB_USER = "YOUR_USERNAME"  # <-- REPLACE THIS!
REPO = "Brain-Tumour-Segmentation"
BRANCH = "main"

files_to_download = {
    'config.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/config.py',
    'scripts/download_data.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/scripts/download_data.py',
    'scripts/__init__.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/scripts/__init__.py',
    'training/train.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/training/train.py',
    'training/metrics.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/training/metrics.py',
    'training/__init__.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/training/__init__.py',
    'models/unet_attention.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/models/unet_attention.py',
    'models/loss_functions.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/models/loss_functions.py',
    'models/attention_gates.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/models/attention_gates.py',
    'models/__init__.py': f'https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/models/__init__.py',
}

print(f"\nDownloading from: {GITHUB_USER}/{REPO}\n")

downloaded = 0
failed = 0

for file_path, url in files_to_download.items():
    # Create directory if file is in a subdirectory
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    result = subprocess.run(['wget', '-q', url, '-O', file_path], capture_output=True)

    if result.returncode == 0:
        print(f"Downloaded: {file_path}")
        downloaded += 1
    else:
        print(f"ERROR: Could not download {file_path}")
        failed += 1

print(f"\nSummary: {downloaded} files downloaded, {failed} failed")
print(f"Working directory: {os.getcwd()}")
```

**IMPORTANT:** Replace `YOUR_USERNAME` with your actual GitHub username!

---

## CELL 4: Install Dependencies

```python
print("="*60)
print("CELL 4: Install Dependencies")
print("="*60)

print("\nInstalling PyTorch 2.4+ (NumPy 2.0 compatible)...")
!pip install -q --upgrade torch torchvision

print("Installing NumPy 2.0+...")
!pip install -q 'numpy>=2.0.0'

print("Installing ML libraries...")
!pip install -q scipy pandas scikit-learn

print("Installing visualization and data libraries...")
!pip install -q matplotlib pillow nibabel

print("Installing training tools...")
!pip install -q pytorch-lightning tensorboard tqdm

print("Installing utilities...")
!pip install -q jupyter ipython

print("\nVerifying installations...")
import numpy as np
import torch
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print("All dependencies installed successfully!")
```

---

## CELL 5: Create Mock Dataset

```python
import os

print("="*60)
print("CELL 5: Create Mock Dataset")
print("="*60)

print(f"\nWorking directory: {os.getcwd()}")
print(f"Config exists: {os.path.exists('config.py')}")
print(f"download_data.py exists: {os.path.exists('scripts/download_data.py')}")

if not os.path.exists('scripts/download_data.py'):
    print("ERROR: download_data.py not found!")
else:
    print("\nGenerating mock BraTS dataset (20 cases)...")
    print("This takes 2-3 minutes...\n")

    !python scripts/download_data.py --create_mock --num_cases 20 --data_dir data/BraTS_MOCK

    print("\nDataset creation complete!")
    print(f"Dataset directory exists: {os.path.exists('data/BraTS_MOCK')}")
```

---

## CELL 6: Configure for Colab

```python
import os

print("="*60)
print("CELL 6: Configure for Colab")
print("="*60)

config_file = 'config.py'

if not os.path.exists(config_file):
    print(f"ERROR: {config_file} not found!")
else:
    with open(config_file, 'r') as f:
        config = f.read()

    # Optimize for Colab T4 GPU
    config = config.replace('BATCH_SIZE = 16', 'BATCH_SIZE = 8')
    config = config.replace('NUM_EPOCHS = 100', 'NUM_EPOCHS = 30')
    config = config.replace('NUM_WORKERS = 4', 'NUM_WORKERS = 2')
    config = config.replace('EARLY_STOPPING_PATIENCE = 20', 'EARLY_STOPPING_PATIENCE = 5')

    with open(config_file, 'w') as f:
        f.write(config)

    print("Configuration updated for Colab:")
    print("  BATCH_SIZE: 8 (reduced from 16)")
    print("  NUM_EPOCHS: 30 (reduced from 100)")
    print("  NUM_WORKERS: 2 (reduced from 4)")
    print("  EARLY_STOPPING_PATIENCE: 5 (reduced from 20)")
```

---

## CELL 7: Start Training

```python
import os
import sys

print("="*60)
print("CELL 7: START TRAINING")
print("="*60)

print(f"\nWorking directory: {os.getcwd()}")
print(f"Dataset exists: {os.path.exists('data/BraTS_MOCK')}")
print(f"Config exists: {os.path.exists('config.py')}")
print(f"Training script exists: {os.path.exists('training/train.py')}")

# Fix Python path for imports
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("\nEstimated time: 1-2 hours for 30 epochs on T4 GPU")
print("="*60)

!python training/train.py \
    --experiment_name colab_v1 \
    --data_dir data/BraTS_MOCK \
    --epochs 30 \
    --batch_size 8 \
    --device cuda \
    --log_dir outputs/logs

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
```

---

## CELL 8: Save Results and Verify

```python
import shutil
import os
import json

print("="*60)
print("CELL 8: Save Results to Google Drive")
print("="*60)

drive_dir = '/content/drive/MyDrive/brain-tumor-training'

# Save trained model
if os.path.exists('checkpoints/colab_v1_best.pth'):
    shutil.copy('checkpoints/colab_v1_best.pth',
                f'{drive_dir}/colab_v1_best.pth')
    print("Model saved to Google Drive")

# Save training summary
if os.path.exists('outputs/colab_v1_summary.json'):
    shutil.copy('outputs/colab_v1_summary.json',
                f'{drive_dir}/colab_v1_summary.json')
    print("Summary saved to Google Drive")

    # Show results
    with open('outputs/colab_v1_summary.json', 'r') as f:
        summary = json.load(f)

    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    if 'best_dice' in summary:
        print(f"Best Dice Score: {summary['best_dice']:.4f}")
    if 'best_iou' in summary:
        print(f"Best IoU Score: {summary['best_iou']:.4f}")
    if 'best_f1' in summary:
        print(f"Best F1 Score: {summary['best_f1']:.4f}")

print("\nAll files saved to Google Drive!")
print(f"Location: {drive_dir}/")
```

---

## Timeline

| Cell | Description       | Time      |
| ---- | ----------------- | --------- |
| 1    | Setup GPU & Drive | 1 min     |
| 2    | Create folders    | 30 sec    |
| 3    | Download files    | 2 min     |
| 4    | Install packages  | 3-5 min   |
| 5    | Create dataset    | 2-3 min   |
| 6    | Configure         | 30 sec    |
| 7    | **Train model**   | 1-2 hours |
| 8    | Save results      | 1 min     |

**Total: Approximately 2-3 hours (mostly training time)**

---

## What Gets Created

After running all cells, you get:

1. **Trained Model** (checkpoints/colab_v1_best.pth)
   - Neural network weights
   - Ready for inference

2. **Training Summary** (outputs/colab_v1_summary.json)
   - Final metrics (Dice, IoU, F1)
   - Loss curves
   - Training history

3. **TensorBoard Logs** (outputs/logs/)
   - Visualization of training progress

All files saved to Google Drive automatically.

---

## After Training: Next Steps

1. **Download Model**
   - Go to Google Drive folder
   - Right-click colab_v1_best.pth
   - Select "Download"

2. **Run Inference**

   ```bash
   python -m uvicorn api.main:app --reload
   ```

   Then visit: http://localhost:8000/docs

3. **Use Web UI**

   ```bash
   cd ui
   npm install
   npm start
   ```

4. **Deploy to Cloud**
   - See docs/CLOUD_DEPLOYMENT.md

---

## Troubleshooting

**CELL 3 fails (files not downloading):**

- Check you replaced YOUR_USERNAME with your actual GitHub username
- Make sure repository is PUBLIC on GitHub

**CELL 4 fails (pip install error):**

- Run: `!pip install --upgrade pip`
- Then re-run CELL 4

**CELL 5 fails (dataset not created):**

- Check: `!ls scripts/`
- Verify download_data.py exists

**CELL 7 fails (training doesn't start):**

- Check: `!ls data/BraTS_MOCK/`
- Verify dataset was created in CELL 5

**Out of memory error:**

- Reduce BATCH_SIZE to 4 in CELL 6
- Reduce NUM_EPOCHS to 20

---

## Get Help

All documentation available in:

- `COLAB_FINAL_SETUP.md` - Detailed Colab guide
- `docs/TRAINING_GUIDE.md` - Complete training documentation
- `docs/ARCHITECTURE.md` - Model architecture details
- `docs/API.md` - API usage

---

## Summary

**You now have:**

- A fully trained brain tumor segmentation neural network
- Checkpoints saved to Google Drive
- Performance metrics and results
- Ready-to-use model for inference

**Without needing:**

- Local GPU
- GPU drivers
- Large storage space
- Installation hassles

Everything runs in free Google Colab!
