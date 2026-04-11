# Google Colab Quick Start 🚀

**No GPU? No problem! Train in Google Colab for FREE**

---

## Quick Setup (5 minutes)

### 1️⃣ Open Google Colab
```
https://colab.research.google.com/
```

### 2️⃣ Enable GPU
Menu: **Runtime → Change runtime type → GPU** (T4/P100/V100)

### 3️⃣ Copy-Paste This (One Cell)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import os
os.chdir('/content/drive/MyDrive')

# Clone repo
!git clone https://github.com/YOUR_USERNAME/Brain-Tumour-Segmentation.git
%cd Brain-Tumour-Segmentation

# Install
!pip install -q torch nibabel numpy scipy pandas scikit-learn matplotlib pillow fastapi uvicorn tensorboard

# Create mock data
!python scripts/download_data.py --create_mock --num_cases 20 --output_dir data/BraTS_MOCK

# Configure for Colab
with open('config.py', 'r') as f:
    config = f.read()
config = config.replace('BATCH_SIZE = 16', 'BATCH_SIZE = 8')
config = config.replace('NUM_EPOCHS = 100', 'NUM_EPOCHS = 30')
with open('config.py', 'w') as f:
    f.write(config)

# Train!
!python training/train.py --experiment_name colab_v1 --epochs 30 --batch_size 8 --device cuda

print("✅ Training Complete! Check Google Drive for saved model.")
```

### 4️⃣ Wait for Training (~1-2 hours for 30 epochs)

### 5️⃣ Download Model
Your trained model is automatically saved to Google Drive!

---

## Hardware Speed Comparison

| GPU | Epochs/Hour | Time for 30 Epochs |
|-----|-------------|----------|
| **T4** (Free) | 10-15 | 2-3 hours |
| **P100** | 20-30 | 1-1.5 hours |
| **V100** | 30-40 | 45-60 min |
| **A100** (Pro) | 40-60 | 30-45 min |

**Free tier = T4 GPU (perfectly fine for training!)**

---

## Full Detailed Guide

📖 See: **`docs/COLAB_SETUP.md`** for step-by-step instructions

---

## Why Colab?

✅ **Free GPU** (T4 GPU = $0)  
✅ **Unlimited storage** (via Google Drive)  
✅ **No installation needed**  
✅ **Pre-installed Python & CUDA**  
✅ **Perfect for students/researchers**  

---

## Replace `YOUR_USERNAME` with your GitHub username

Then just run and wait! Easy. 🎉
