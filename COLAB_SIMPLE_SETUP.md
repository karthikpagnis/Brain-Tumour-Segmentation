# Colab Simple Setup - When Repository Path Issues Occur

If you get "can't open file scripts/download_data.py" error, use this simplified approach:

## Alternative Method for Colab

### CELL 1: Basic Setup
```python
import torch
from google.colab import drive
import os

# Check GPU
print(f"GPU: {torch.cuda.is_available()}")

# Mount Drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive')
```

### CELL 2: Download using curl (Alternative)
```python
import os
os.makedirs('brain-tumor-workspace', exist_ok=True)
os.chdir('brain-tumor-workspace')

# Download specific files directly instead of cloning entire repo
!mkdir -p Brain-Tumour-Segmentation/scripts
!mkdir -p Brain-Tumour-Segmentation/training
!mkdir -p Brain-Tumour-Segmentation/models
!mkdir -p Brain-Tumour-Segmentation/data
!mkdir -p Brain-Tumour-Segmentation/outputs
!mkdir -p Brain-Tumour-Segmentation/checkpoints

# Download key files
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/config.py -O Brain-Tumour-Segmentation/config.py
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/scripts/download_data.py -O Brain-Tumour-Segmentation/scripts/download_data.py
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/training/train.py -O Brain-Tumour-Segmentation/training/train.py
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/Brain-Tumour-Segmentation/main/requirements.txt -O Brain-Tumour-Segmentation/requirements.txt

os.chdir('Brain-Tumour-Segmentation')
```

### CELL 3: Install & Create Dataset
```python
!pip install -q -r requirements.txt

# Now scripts/download_data.py should exist
# Important: Use --data_dir (NOT --output_dir)
!python scripts/download_data.py --create_mock --num_cases 20 --data_dir data/BraTS_MOCK
```

### CELL 4: Configure & Train
```python
with open('config.py', 'r') as f:
    config = f.read()

config = config.replace('BATCH_SIZE = 16', 'BATCH_SIZE = 8')
config = config.replace('NUM_EPOCHS = 100', 'NUM_EPOCHS = 30')

with open('config.py', 'w') as f:
    f.write(config)

!python training/train.py --experiment_name colab_v1 --epochs 30 --batch_size 8 --device cuda
```

## Why This Works

1. Uses wget to download specific files (more reliable)
2. Pre-creates directory structure
3. No complex git cloning needed
4. Files are guaranteed to exist before running

## Before Running

Replace YOUR_USERNAME with your GitHub username wherever it appears.
