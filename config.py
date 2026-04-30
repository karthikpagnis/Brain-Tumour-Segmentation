"""
Configuration File for Brain Tumor Segmentation Project
All hyperparameters, model settings, and training configurations
"""

import torch
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEVICE CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL ARCHITECTURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Input/Output
NUM_INPUT_CHANNELS = 4  # T1, T1ce, T2, FLAIR
NUM_CLASSES = 4  # Background, Necrotic Core, Edema, Enhancing Tumor

# Encoder-Decoder Channels
ENCODER_CHANNELS = [32, 64, 128, 256]  # Progressive channel expansion
DECODER_CHANNELS = [256, 128, 64, 32]  # Progressive channel contraction

# Architecture Features
USE_ATTENTION_GATES = True  # Use spatial attention mechanisms
USE_BATCH_NORM = True  # Batch normalization in convolutions
DROPOUT_RATE = 0.2  # Dropout probability for regularization

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-5
WARMUP_EPOCHS = 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OPTIMIZER CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTIMIZER = "adam"  # Options: "adam", "sgd"
MOMENTUM = 0.9  # For SGD optimizer
WEIGHT_DECAY = 1e-4  # L2 regularization

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEARNING RATE SCHEDULER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCHEDULER = "reduce_on_plateau"  # Options: "reduce_on_plateau", "cosine"
SCHEDULER_FACTOR = 0.1  # Multiply lr by this factor
SCHEDULER_PATIENCE = 5  # Epochs to wait before reducing lr

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOSS FUNCTION CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DICE_LOSS_WEIGHT = 0.5  # Weight of Dice loss in composite loss
BCE_LOSS_WEIGHT = 0.5   # Weight of BCE loss in composite loss
DICE_SMOOTH = 1e-7      # Smoothing constant for Dice loss

# Class weights for handling class imbalance (1.0 = equal weight)
CLASS_WEIGHTS = torch.tensor([
    0.1,    # Background (downweight due to dominance)
    2.0,    # Necrotic Core (rare, upweight)
    1.5,    # Edema (upweight)
    2.0,    # Enhancing Tumor (rare, upweight)
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Input data shape (voxels)
INPUT_SHAPE = (240, 240, 155)

# Data normalization
NORMALIZE_DATA = True
NORMALIZE_METHOD = "znorm"  # Options: "znorm" (z-score), "minmax"

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5  # Probability of applying augmentation

# Augmentation parameters
ROTATION_ANGLES = (-15, 15)  # Degrees
ELASTIC_DEFORM_SIGMA = 5
ELASTIC_DEFORM_ALPHA = 50
FLIP_AXES = (0, 1, 2)  # Flip along all axes
BRIGHTNESS_RANGE = (0.9, 1.1)
CONTRAST_RANGE = (0.9, 1.1)
GAMMA_RANGE = (0.8, 1.2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EARLY STOPPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EARLY_STOPPING_PATIENCE = 20  # Epochs to wait before stopping
EARLY_STOPPING_METRIC = "dice"  # Options: "dice", "loss"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIRECTORIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
TENSORBOARD_LOG_DIR = OUTPUTS_DIR / "logs"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
for directory in [CHECKPOINTS_DIR, OUTPUTS_DIR, TENSORBOARD_LOG_DIR, UPLOADS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPERIMENT_NAME = "attention_unet_brats_v1"
USE_TENSORBOARD = True
SAVE_BEST_MODEL = True
SAVE_CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METRICS_TO_TRACK = ["dice", "iou", "f1", "sensitivity", "specificity"]
DICE_THRESHOLD = 0.5  # Classification threshold for metrics

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOG_LEVEL = "INFO"
LOG_FILE = OUTPUTS_DIR / "training.log"
VERBOSE = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API SERVER CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4
API_RELOAD = False  # Set to True for development

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INFERENCE CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_CHECKPOINT = CHECKPOINTS_DIR / "best_model.pth"
INFERENCE_BATCH_SIZE = 8
INFERENCE_THRESHOLD = 0.5
POSTPROCESS_THRESHOLD = 10  # Minimum number of voxels to keep a connected component

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLASS LABELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASS_LABELS = {
    0: "Background",
    1: "Necrotic Core",
    2: "Peritumoral Edema",
    3: "Enhancing Tumor",
}

CLASS_COLORS = {
    0: (0, 0, 0),          # Black - Background
    1: (255, 0, 0),        # Red - Necrotic Core
    2: (0, 255, 0),        # Green - Edema
    3: (0, 0, 255),        # Blue - Enhancing Tumor
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEBUGGING & DEVELOPMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEBUG_MODE = False
SEED = 42  # For reproducibility
DRY_RUN = False  # If True, only process first batch

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Print Configuration Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 80)
    print("BRAIN TUMOR SEGMENTATION - CONFIGURATION")
    print("=" * 80)
    print(f"\n📱 DEVICE: {DEVICE}")
    print(f"🧠 MODEL: Attention-Enhanced 3D U-Net")
    print(f"  - Input Channels: {NUM_INPUT_CHANNELS}")
    print(f"  - Output Classes: {NUM_CLASSES}")
    print(f"  - Encoder Channels: {ENCODER_CHANNELS}")
    print(f"  - Attention Gates: {USE_ATTENTION_GATES}")
    print(f"\n⚙️  TRAINING:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Optimizer: {OPTIMIZER.upper()}")
    print(f"  - Loss: Dice-BCE ({DICE_LOSS_WEIGHT:.1%} / {BCE_LOSS_WEIGHT:.1%})")
    print(f"\n📂 DIRECTORIES:")
    print(f"  - Data: {DATA_DIR}")
    print(f"  - Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  - Outputs: {OUTPUTS_DIR}")
    print("=" * 80 + "\n")
