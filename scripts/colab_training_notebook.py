"""
Google Colab Training Script - Copy this into Colab Notebook cells

Each code block is a separate Colab cell. Just copy-paste and run!
"""

# ============================================================================
# CELL 1: Check GPU & Mount Drive
# ============================================================================
print("STEP 1: Checking GPU...")
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")

print("\nMounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print("✓ Drive mounted!")

# ============================================================================
# CELL 2: Clone Repository & Install
# ============================================================================
import os
import subprocess

print("STEP 2: Setting up repository...\n")

# Create working directory
os.makedirs('/content/drive/MyDrive/brain-tumor-workspace', exist_ok=True)
os.chdir('/content/drive/MyDrive/brain-tumor-workspace')

# Clone repo (only if not already cloned)
if not os.path.exists('Brain-Tumour-Segmentation'):
    print("Cloning repository...")
    !git clone https://github.com/YOUR_USERNAME/Brain-Tumour-Segmentation.git
    print("✓ Repository cloned!")
else:
    print("✓ Repository already cloned")

# Navigate to project
os.chdir('Brain-Tumour-Segmentation')
print(f"Working directory: {os.getcwd()}\n")

# Install dependencies (suppress output with -q)
print("Installing dependencies (this may take 2-3 minutes)...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q nibabel numpy scipy pandas scikit-learn matplotlib pillow
!pip install -q fastapi uvicorn pytest tensorboard

print("✓ Dependencies installed!")

# ============================================================================
# CELL 3: Create Mock Dataset (for quick testing)
# ============================================================================
print("STEP 3: Creating mock BraTS dataset...")
!python scripts/download_data.py --create_mock --num_cases 20 --output_dir data/BraTS_MOCK

print("\n✓ Mock dataset created with 20 cases (~500MB)")
print("Dataset location: data/BraTS_MOCK/")

# Verify dataset
import os
dataset_path = 'data/BraTS_MOCK/BraTS2021_Training'
if os.path.exists(dataset_path):
    num_cases = len(os.listdir(dataset_path))
    print(f"✓ Verified: {num_cases} training cases found")

# ============================================================================
# CELL 4: Configure for Colab Hardware
# ============================================================================
print("STEP 4: Optimizing config for Colab...\n")

# Read current config
with open('config.py', 'r') as f:
    config_content = f.read()

# Original settings to replace
replacements = {
    'BATCH_SIZE = 16': 'BATCH_SIZE = 8',
    'NUM_EPOCHS = 100': 'NUM_EPOCHS = 30',
    'NUM_WORKERS = 4': 'NUM_WORKERS = 2',
    'EARLY_STOPPING_PATIENCE = 20': 'EARLY_STOPPING_PATIENCE = 5',
}

# Apply replacements
for old, new in replacements.items():
    if old in config_content:
        config_content = config_content.replace(old, new)
        print(f"✓ Changed: {old} → {new}")

# Write updated config
with open('config.py', 'w') as f:
    f.write(config_content)

print("\n✓ Config optimized for Colab free tier GPU")
print("  - Batch size: 8 (memory efficient)")
print("  - Epochs: 30 (quick training)")
print("  - Workers: 2 (stable on Colab)")

# ============================================================================
# CELL 5: Start Training
# ============================================================================
print("STEP 5: Starting training...\n")
print("=" * 60)
print("Training Progress")
print("=" * 60 + "\n")

# Run training
import subprocess
result = subprocess.run([
    'python', 'training/train.py',
    '--experiment_name', 'colab_v1',
    '--data_dir', 'data/BraTS_MOCK',
    '--epochs', '30',
    '--batch_size', '8',
    '--device', 'cuda',
    '--log_dir', 'outputs/logs'
], text=True)

if result.returncode == 0:
    print("\n✓ Training completed successfully!")
else:
    print("\n✗ Training ended. Check error messages above.")

# ============================================================================
# CELL 6: View Training Progress (TensorBoard)
# ============================================================================
print("STEP 6: Viewing training progress with TensorBoard...\n")

# Load TensorBoard extension
%load_ext tensorboard

# Launch TensorBoard
%tensorboard --logdir outputs/logs/tensorboard

print("TensorBoard should appear above!")
print("Showing: Loss curves, Validation metrics, Training progress")

# ============================================================================
# CELL 7: Wait for Training & Save Checkpoints
# ============================================================================
print("STEP 7: Saving results to Google Drive...\n")

import shutil
import os
import time

# Wait for training to finish
while not os.path.exists('checkpoints/colab_v1_best.pth'):
    print("Waiting for training to finish...")
    time.sleep(30)

print("✓ Training finished!")

# Save best model to Drive
source_model = 'checkpoints/colab_v1_best.pth'
dest_model = '/content/drive/MyDrive/brain-tumor-workspace/colab_v1_best.pth'

shutil.copy(source_model, dest_model)
print(f"✓ Model saved to Drive: {dest_model}")

# Save training summary
source_summary = 'outputs/colab_v1_summary.json'
dest_summary = '/content/drive/MyDrive/brain-tumor-workspace/colab_v1_summary.json'

if os.path.exists(source_summary):
    shutil.copy(source_summary, dest_summary)
    print(f"✓ Summary saved to Drive: {dest_summary}")

print("\nFiles saved successfully!")
print("You can now download them from Google Drive!")

# ============================================================================
# CELL 8: Load & Evaluate Model
# ============================================================================
print("STEP 8: Loading and evaluating model...\n")

import torch
from models.unet_attention import AttentionUNet3D

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = AttentionUNet3D().to(device)
checkpoint = torch.load('checkpoints/colab_v1_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded!")

# Test inference
print("\nTesting inference...")
dummy_input = torch.randn(1, 4, 32, 32, 32).to(device)

with torch.no_grad():
    output = model(dummy_input)

print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print("✓ Inference test passed!")

# Print model info
num_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Parameters: {num_params / 1e6:.1f}M")
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================================
# CELL 9: Download Results
# ============================================================================
print("STEP 9: Download results to local machine...\n")

from google.colab import files
import os

# Create files to download
download_files = [
    'checkpoints/colab_v1_best.pth',
    'outputs/colab_v1_summary.json',
    'outputs/training.log'
]

for file in download_files:
    if os.path.exists(file):
        print(f"Downloading {file}...")
        files.download(file)
        print(f"✓ Downloaded!")

print("\nAll files downloaded! Check your Downloads folder.")

# ============================================================================
# CELL 10: View Results Summary
# ============================================================================
print("STEP 10: Viewing results summary...\n")

import json

if os.path.exists('outputs/colab_v1_summary.json'):
    with open('outputs/colab_v1_summary.json', 'r') as f:
        summary = json.load(f)

    print("✓ TRAINING RESULTS")
    print("=" * 60)
    print(f"Epochs Trained:     {summary.get('epoch', 'N/A')}")
    print(f"Final Train Loss:   {summary.get('train_loss', [-1])[-1]:.4f}")
    print(f"Final Val Loss:     {summary.get('val_loss', [-1])[-1]:.4f}")
    print(f"Best Val Dice:      {summary.get('best_dice', 'N/A'):.4f}")
    print(f"Best Val IoU:       {summary.get('best_iou', 'N/A'):.4f}")
    print(f"Best Val F1:        {summary.get('best_f1', 'N/A'):.4f}")
    print("=" * 60)
else:
    print("Summary not found. Training may still be in progress.")

print("\n✅ YOU'RE DONE!")
print("\nNext steps:")
print("1. Download model from Google Drive or Downloads folder")
print("2. Use FastAPI to deploy: python -m uvicorn api.main:app")
print("3. Try web UI: cd ui && npm install && npm start")
print("4. See docs/ for advanced features and deployment options")
