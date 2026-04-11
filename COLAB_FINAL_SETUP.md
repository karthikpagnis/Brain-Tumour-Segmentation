# Google Colab - Final Working Setup

Complete working setup that avoids all pip dependency issues.

---

## Step 1: Open Google Colab

Go to: https://colab.research.google.com/

---

## Step 2: Enable GPU

Menu: **Runtime → Change runtime type → GPU**
Select: **T4** (free)

---

## Step 3: Copy Each Cell Into Colab

Run cells one by one. Press `Shift + Enter` after each cell.

---

## CELL 1: Mount Drive and Setup

```python
import torch
from google.colab import drive
import os

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Mount Drive
drive.mount('/content/drive')

# Create workspace
workspace = '/content/drive/MyDrive/brain-tumor'
os.makedirs(workspace, exist_ok=True)
os.chdir(workspace)

print(f"Working directory: {os.getcwd()}")
```

---

## CELL 2: Create Project Structure

```python
import os

os.makedirs('Brain-Tumour-Segmentation/scripts', exist_ok=True)
os.makedirs('Brain-Tumour-Segmentation/training', exist_ok=True)
os.makedirs('Brain-Tumour-Segmentation/models', exist_ok=True)
os.makedirs('Brain-Tumour-Segmentation/data', exist_ok=True)
os.makedirs('Brain-Tumour-Segmentation/outputs', exist_ok=True)
os.makedirs('Brain-Tumour-Segmentation/checkpoints', exist_ok=True)

os.chdir('Brain-Tumour-Segmentation')

print(f"Current directory: {os.getcwd()}")
print(f"Directories created: {os.listdir('.')}")
```

---

## CELL 3: Download Required Files

```python
import os
import subprocess

files_to_download = {
    'config.py': 'https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/config.py',
    'scripts/download_data.py': 'https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/scripts/download_data.py',
    'training/train.py': 'https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/training/train.py',
    'models/unet_attention.py': 'https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/models/unet_attention.py',
    'models/loss_functions.py': 'https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/models/loss_functions.py',
    'training/metrics.py': 'https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/training/metrics.py',
}

print("Downloading files...")
for file_path, url in files_to_download.items():
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    result = subprocess.run(['wget', '-q', url, '-O', file_path], capture_output=True)
    if result.returncode == 0:
        print(f"Downloaded: {file_path}")
    else:
        print(f"ERROR downloading {file_path}")

print("\nFile check:")
print(f"config.py exists: {os.path.exists('config.py')}")
print(f"download_data.py exists: {os.path.exists('scripts/download_data.py')}")
print(f"train.py exists: {os.path.exists('training/train.py')}")
```

**IMPORTANT: Replace YOUR_USERNAME with your GitHub username**

---

## CELL 4: Install PyTorch and Dependencies

```python
# Install PyTorch (Colab compatible)
!pip install -q torch==2.2.0 torchvision==0.17.0

# Install ML libraries
!pip install -q numpy scipy pandas scikit-learn matplotlib pillow

# Install training tools
!pip install -q pytorch-lightning tensorboard tqdm

# Install other utilities
!pip install -q nibabel jupyter ipython

print("All packages installed successfully")
```

---

## CELL 5: Create Mock Dataset

```python
import os

print("Creating mock BraTS dataset...")
print(f"Current directory: {os.getcwd()}")
print(f"Scripts directory exists: {os.path.exists('scripts')}")
print(f"download_data.py exists: {os.path.exists('scripts/download_data.py')}")

!python scripts/download_data.py --create_mock --num_cases 20 --data_dir data/BraTS_MOCK

print("\nDataset created!")
print(f"Dataset directory: {os.path.exists('data/BraTS_MOCK')}")
```

---

## CELL 6: Configure for Colab

```python
print("Configuring for Colab...")

with open('config.py', 'r') as f:
    config = f.read()

config = config.replace('BATCH_SIZE = 16', 'BATCH_SIZE = 8')
config = config.replace('NUM_EPOCHS = 100', 'NUM_EPOCHS = 30')
config = config.replace('NUM_WORKERS = 4', 'NUM_WORKERS = 2')
config = config.replace('EARLY_STOPPING_PATIENCE = 20', 'EARLY_STOPPING_PATIENCE = 5')

with open('config.py', 'w') as f:
    f.write(config)

print("Configuration updated:")
print("  - BATCH_SIZE: 8 (memory efficient)")
print("  - NUM_EPOCHS: 30 (quick training)")
print("  - NUM_WORKERS: 2 (stable)")
print("  - EARLY_STOPPING_PATIENCE: 5")
```

---

## CELL 7: Start Training

```python
import os

print("="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Current directory: {os.getcwd()}")
print(f"Dataset exists: {os.path.exists('data/BraTS_MOCK')}")
print(f"Config exists: {os.path.exists('config.py')}")

!python training/train.py \
    --experiment_name colab_v1 \
    --data_dir data/BraTS_MOCK \
    --epochs 30 \
    --batch_size 8 \
    --device cuda \
    --log_dir outputs/logs

print("\nTraining completed!")
```

---

## CELL 8: Save Results to Drive

```python
import shutil
import os

print("Saving results to Google Drive...")

drive_dir = '/content/drive/MyDrive/brain-tumor'

if os.path.exists('checkpoints/colab_v1_best.pth'):
    shutil.copy('checkpoints/colab_v1_best.pth', f'{drive_dir}/colab_v1_best.pth')
    print("Model saved")

if os.path.exists('outputs/colab_v1_summary.json'):
    shutil.copy('outputs/colab_v1_summary.json', f'{drive_dir}/colab_v1_summary.json')
    print("Summary saved")

print("All files saved to Google Drive!")
```

---

## Summary

| Step | Description | Time |
|------|-------------|------|
| 1-2 | Setup and create directories | 1 min |
| 3 | Download files | 2 min |
| 4 | Install packages | 3-5 min |
| 5 | Create dataset | 2 min |
| 6 | Configure | 30 sec |
| 7 | Train model | 1-2 hours |
| 8 | Save results | 1 min |

**Total: Approximately 2-3 hours including training**

---

## Troubleshooting

If CELL 5 (dataset creation) fails:
- Make sure YOUR_USERNAME is replaced in CELL 3
- Verify download_data.py was downloaded: `!ls scripts/`

If CELL 7 (training) fails:
- Check dataset exists: `!ls data/BraTS_MOCK/`
- Check config.py exists: `!cat config.py | head -20`

If you run out of memory:
- Reduce BATCH_SIZE to 4 in CELL 6
- Reduce num_cases to 10 in CELL 5

---

## Next Steps After Training

1. Download model from Google Drive to your local machine
2. Try inference with FastAPI: `python -m uvicorn api.main:app`
3. See docs/ for advanced features and deployment

---

## Before You Start

Replace `YOUR_USERNAME` with your actual GitHub username in CELL 3.
