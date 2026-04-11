"""
Configuration file for Brain Tumour Segmentation project
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# BraTS Dataset
BRATS_VERSION = "2021"  # Can be updated to newer versions
DATASET_NAME = "BraTS"

# Data split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# MRI Modalities
MRI_MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]
NUM_INPUT_CHANNELS = len(MRI_MODALITIES)

# Tumor classes
TUMOR_CLASSES = {
    0: "Background",
    1: "Necrotic Core",
    2: "Peritumoral Edema",
    3: "Enhancing Tumor"
}
NUM_CLASSES = len(TUMOR_CLASSES)

# Data preprocessing
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
IMAGE_DEPTH = 155  # Number of slices in z-axis

# Normalization
NORMALIZATION_METHOD = "zscore"  # "zscore" or "minmax"
INTENSITY_CLIP_RANGE = (-1.0, 5.0)  # After normalization

# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================

AUGMENTATION_SETTINGS = {
    "rotate_range": (-15, 15),  # degrees
    "shift_range": (-0.1, 0.1),  # fraction of image size
    "zoom_range": (0.9, 1.1),
    "horizontal_flip": True,
    "vertical_flip": True,
    "elastic_deformation": True,
    "elastic_alpha": (30, 30),
    "elastic_sigma": (3, 3),
    "intensity_shifts": True,
    "intensity_shift_range": (-0.2, 0.2),
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
    "gamma_range": (0.8, 1.2),
    "noise_std": 0.01,
}

# Augmentation probability
AUGMENTATION_PROBABILITY = 0.5

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# U-Net Architecture
ENCODER_CHANNELS = [NUM_INPUT_CHANNELS, 32, 64, 128, 256]
DECODER_CHANNELS = [256, 128, 64, 32, NUM_INPUT_CHANNELS]

# Attention Gate Configuration
USE_ATTENTION_GATES = True
ATTENTION_TYPE = "channel"  # "channel", "spatial", or "hybrid"

# Batch Normalization
USE_BATCH_NORM = True
DROPOUT_RATE = 0.2

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Device
DEVICE = "cuda"  # "cuda", "mps" (Mac), or "cpu"

# Batch size
BATCH_SIZE = 16  # Recommended: 16-32 for medical imaging
NUM_WORKERS = 4

# Learning rate and optimizer
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9

# Optimizer choice
OPTIMIZER = "adam"  # "adam", "sgd", or "adamw"

# Learning rate scheduler
SCHEDULER = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step"
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# Training epochs
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20

# Loss function weights
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT = 0.5
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# Gradient clipping
GRADIENT_CLIP_VALUE = 1.0

# Mixed precision training
USE_AMP = False  # Automatic Mixed Precision

# ============================================================================
# VALIDATION & EVALUATION
# ============================================================================

# Evaluation metrics
METRICS = ["dice", "iou", "f1_score"]

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.5

# Post-processing
USE_MORPHOLOGICAL_OPERATIONS = True
MORPHOLOGICAL_KERNEL_SIZE = 3

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

# Inference settings
INFERENCE_BATCH_SIZE = 8
INFERENCE_THRESHOLD = 0.5

# ============================================================================
# API CONFIGURATION
# ============================================================================

# FastAPI settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4
API_DEBUG = False

# CORS settings
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

# File upload settings
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = [".nii", ".nii.gz", ".nifti"]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging level
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

# TensorBoard
USE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

# Random seeds
RANDOM_SEED = 42
NUMPY_SEED = 42
TORCH_SEED = 42

# Deterministic behavior
TORCH_DETERMINISTIC = True
TORCH_BENCHMARK = False

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Experiment name and description
EXPERIMENT_NAME = "attention_unet_v1"
EXPERIMENT_DESCRIPTION = "Attention-Enhanced U-Net for Brain Tumor Segmentation"

# Save checkpoints
SAVE_BEST_MODEL = True
SAVE_CHECKPOINT_EVERY_N_EPOCHS = 5
KEEP_LAST_N_CHECKPOINTS = 3

# ============================================================================
# RESEARCH CONFIGURATION
# ============================================================================

# Baseline comparison
RUN_BASELINE = True
BASELINE_MODEL = "unet_standard"

# Ablation studies
RUN_ABLATION_STUDIES = True
ABLATION_CONFIGS = {
    "no_attention": {"USE_ATTENTION_GATES": False},
    "dice_only": {"DICE_LOSS_WEIGHT": 1.0, "BCE_LOSS_WEIGHT": 0.0},
    "bce_only": {"DICE_LOSS_WEIGHT": 0.0, "BCE_LOSS_WEIGHT": 1.0},
}

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Validation frequency (run validation every N batches)
VALIDATION_FREQUENCY = 100

# Print training metrics every N batches
PRINT_FREQUENCY = 50

# ============================================================================
# Function to get config as dictionary
# ============================================================================

def get_config_dict():
    """Return configuration as dictionary"""
    return {
        k: v
        for k, v in globals().items()
        if not k.startswith("_") and k.isupper() and not callable(v)
    }

if __name__ == "__main__":
    # Print configuration
    print("Brain Tumour Segmentation - Configuration")
    print("=" * 80)
    config = get_config_dict()
    for key, value in sorted(config.items()):
        print(f"{key}: {value}")
