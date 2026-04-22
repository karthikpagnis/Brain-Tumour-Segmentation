# Brain Tumor Segmentation Project - Complete Documentation

## 🎯 PROJECT OVERVIEW

**Name:** Brain Tumor Segmentation from MRI Scans using Attention-Enhanced U-Net
**Purpose:** Automated segmentation of brain tumors (4 classes) from multimodal MRI scans
**Framework:** PyTorch 2.4.0+
**Python:** 3.9+
**Domain:** Medical Image Segmentation, Deep Learning, Computer Vision

---

## 📊 PROJECT GOALS

1. **Build an AI Model**: Segment brain tumors from MRI scans automatically
2. **Research**: Implement and test Attention-Enhanced U-Net architecture
3. **Accuracy**: Achieve >85% Dice Similarity Coefficient (DSC)
4. **Deployment**: Provide REST API + Web UI for inference
5. **Reproducibility**: Document entire pipeline end-to-end

---

## 📁 FILE STRUCTURE & INTENTIONS

```
Brain-Tumour-Segmentation/
│
├── 📂 data/                          # DATA PIPELINE
│   ├── __init__.py                   # Package initialization
│   ├── preprocessing.py              # NIfTI loading, normalization, resizing
│   │   └── NIfTIPreprocessor class   # Main preprocessing logic
│   ├── augmentation.py               # Spatial & intensity data augmentation
│   │   └── DataAugmentor class       # Random transforms for training
│   └── dataloader.py                 # PyTorch DataLoader
│       └── BraTS2021DataLoader       # Custom dataset with lazy loading
│
├── 📂 models/                        # NEURAL NETWORK ARCHITECTURES
│   ├── __init__.py
│   ├── unet_attention.py             # Main Attention-Enhanced U-Net
│   │   ├── EncoderBlock              # Downsampling blocks (4 levels)
│   │   ├── DecoderBlock              # Upsampling blocks with attention
│   │   ├── BottleneckBlock           # Central feature extraction
│   │   └── AttentionUNet3D           # Full model orchestration
│   ├── attention_gates.py            # Attention mechanisms
│   │   ├── AttentionGate             # Spatial attention module
│   │   ├── ChannelAttention          # Channel-wise attention (SE-Net)
│   │   ├── DoubleConvBlock3D         # Conv→BatchNorm→ReLU pairs
│   │   └── ConvBlock3D               # Single conv block
│   └── loss_functions.py             # Loss functions
│       ├── DiceLoss                  # Dice similarity loss
│       ├── BCELoss                   # Binary cross-entropy loss
│       └── DiceBCELoss               # Composite loss (50%-50%)
│
├── 📂 training/                      # TRAINING PIPELINE
│   ├── __init__.py
│   ├── train.py                      # Main training loop
│   │   └── Trainer class             # Manages training/validation epochs
│   ├── metrics.py                    # Evaluation metrics
│   │   ├── Dice                      # DSC calculation
│   │   ├── IoU                       # Intersection over Union
│   │   ├── F1Score                   # Precision-recall harmonic mean
│   │   └── MetricAggregator          # Batch metric aggregation
│   └── validate.py                   # Validation-only loop
│
├── 📂 inference/                     # PREDICTION PIPELINE
│   ├── __init__.py
│   ├── predict.py                    # Single/batch predictions
│   │   └── ModelPredictor            # Inference wrapper
│   └── postprocess.py                # Output processing
│       └── PostProcessor             # Argmax, threshold, NIfTI save
│
├── 📂 api/                           # REST API (FastAPI)
│   ├── __init__.py
│   ├── main.py                       # FastAPI app setup
│   │   └── app initialization        # Routes, CORS, middleware
│   ├── routes.py                     # API endpoints
│   │   ├── POST /api/predict         # File upload + segmentation
│   │   ├── GET /api/model-info       # Model metadata
│   │   └── GET /api/health           # Health check
│   ├── schemas.py                    # Request/response models (Pydantic)
│   └── utils.py                      # Helper functions
│
├── 📂 ui/                            # REACT WEB UI
│   ├── public/
│   ├── src/
│   │   ├── components/               # React components
│   │   │   ├── ImageUploader.jsx     # Drag-drop file upload
│   │   │   ├── MRIViewer.jsx         # Slice navigation viewer
│   │   │   ├── ResultsDisplay.jsx    # Segmentation visualization
│   │   │   └── MetricsDisplay.jsx    # Performance metrics
│   │   ├── pages/                    # Page components
│   │   ├── App.jsx                   # Main app entry
│   │   └── index.js                  # React DOM render
│   └── package.json                  # NPM dependencies
│
├── 📂 experiments/                   # RESEARCH & ABLATION STUDIES
│   ├── __init__.py
│   ├── baseline_unet.py              # Standard U-Net (no attention)
│   └── analysis.py                   # Results visualization & comparison
│
├── 📂 tests/                         # UNIT & INTEGRATION TESTS
│   ├── __init__.py
│   ├── test_models.py                # Model forward pass tests
│   ├── test_preprocessing.py         # Data loading tests
│   └── test_dataloader.py            # DataLoader tests
│
├── 📂 scripts/                       # UTILITY SCRIPTS
│   ├── download_data.py              # Download BraTS dataset via kagglehub
│   ├── colab_training_notebook.py    # Google Colab training
│   └── evaluate_model.py             # Evaluation on test set
│
├── 📂 docs/                          # DOCUMENTATION
│   ├── ARCHITECTURE.md               # Detailed technical architecture
│   ├── DATASET.md                    # BraTS dataset explanation
│   ├── TRAINING_GUIDE.md             # How to train
│   ├── API.md                        # API documentation
│   ├── COLAB_SETUP.md                # Google Colab instructions
│   ├── DEPLOYMENT.md                 # Production deployment
│   └── RESULTS.md                    # Experimental results
│
├── 📂 checkpoints/                   # SAVED MODEL WEIGHTS
│   ├── best_model.pth                # Best validation checkpoint
│   ├── epoch_50.pth                  # Periodic checkpoints
│   └── final_model.pth               # Training completion checkpoint
│
├── 📂 outputs/                       # TRAINING OUTPUTS
│   ├── logs/                         # TensorBoard logs
│   ├── predictions/                  # Model predictions (NIfTI)
│   └── training.log                  # Training log file
│
├── config.py                         # GLOBAL CONFIGURATION
│   └── All hyperparameters, paths, settings (see below)
│
├── setup.py                          # Package setup for pip install
├── requirements.txt                  # Python dependencies
├── requirements_colab.txt            # Colab-specific dependencies
├── README.md                         # Project overview
├── RUN_GUIDE.md                      # Quick start guide
└── PROJECT_SUMMARY.md                # High-level summary

```

---

## ⚙️ CONFIGURATION (config.py)

All project settings centralized in `config.py`:

### Dataset Configuration

```python
BRATS_VERSION = "2021"
NUM_INPUT_CHANNELS = 4                # T1, T1ce, T2, FLAIR
NUM_CLASSES = 4                       # Background, Necrotic, Edema, Enhancing
IMAGE_SIZE = (240, 240, 155)          # Height × Width × Depth
NORMALIZATION_METHOD = "zscore"       # Z-score or min-max normalization
```

### Model Configuration

```python
ENCODER_CHANNELS = [4, 32, 64, 128, 256]
DECODER_CHANNELS = [256, 128, 64, 32, 4]
USE_ATTENTION_GATES = True            # Enable attention mechanisms
DROPOUT_RATE = 0.2
USE_BATCH_NORM = True
```

### Training Configuration

```python
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
OPTIMIZER = "adam"                    # adam, sgd, adamw
SCHEDULER = "reduce_on_plateau"       # LR scheduling strategy
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT = 0.5
GRADIENT_CLIP_VALUE = 1.0
```

### Data Augmentation

```python
AUGMENTATION_SETTINGS = {
    "rotate_range": (-15, 15),        # Random rotations
    "horizontal_flip": True,
    "vertical_flip": True,
    "elastic_deformation": True,      # Elastic warping
    "intensity_shifts": True,         # Brightness/contrast
    "gamma_range": (0.8, 1.2),        # Non-linear intensity
    "noise_std": 0.01,                # Gaussian noise
}
```

### API Configuration

```python
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE = 100 * 1024 * 1024    # 100 MB limit
ALLOWED_EXTENSIONS = [".nii", ".nii.gz"]
```

---

## 🏗️ ARCHITECTURE OVERVIEW

### System Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: MRI NIfTI File                    │
│              (4 modalities: T1, T1ce, T2, FLAIR)            │
│                   Size: 240×240×155 voxels                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │   DATA PREPROCESSING (data/*)       │
        │   • Load NIfTI file (nibabel)       │
        │   • Z-score normalization           │
        │   • Standardize to 240×240×155      │
        │   • Split train/val/test (80/10/10) │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │   DATA AUGMENTATION (training only) │
        │   • Random 3D rotations (±15°)      │
        │   • Horizontal/vertical flips       │
        │   • Elastic deformations            │
        │   • Intensity augmentation          │
        └──────────────────┬──────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │     ATTENTION-ENHANCED U-NET (models/*)     │
    │                                             │
    │  INPUT: (Batch, 4, 240, 240, 155)          │
    │  ╔════════════════════════════════════════╗│
    │  ║  ENCODER (Downsampling)                ║│
    │  ║  Block 1: 4→32 ch, 240→120 spatial    ║│
    │  ║  Block 2: 32→64 ch, 120→60 spatial    ║│
    │  ║  Block 3: 64→128 ch, 60→30 spatial    ║│
    │  ║  Block 4: 128→256 ch, 30→15 spatial   ║│
    │  ╚════════════════════════════════════════╝│
    │           ↓                                 │
    │  ╔════════════════════════════════════════╗│
    │  ║  BOTTLENECK                            ║│
    │  ║  Double Conv: 256→256 channels         ║│
    │  ╚════════════════════════════════════════╝│
    │           ↓                                 │
    │  ╔════════════════════════════════════════╗│
    │  ║  DECODER (Upsampling) with Attention   ║│
    │  ║  Block 4: 256→128 ch + Attention       ║│
    │  ║  Block 3: 128→64 ch + Attention        ║│
    │  ║  Block 2: 64→32 ch + Attention         ║│
    │  ║  Block 1: 32→4 ch + Attention          ║│
    │  ║                                         ║│
    │  ║  Skip connections from encoder         ║│
    │  ║  Attention gates re-weight skip        ║│
    │  ╚════════════════════════════════════════╝│
    │                                             │
    │  OUTPUT: (Batch, 4, 240, 240, 155)        │
    │  (Logits for 4 classes: BG, Necrotic,     │
    │   Edema, Enhancing)                       │
    └──────────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   LOSS FUNCTION (models/)   │
        │   Dice-BCE Composite Loss   │
        │   L_total = 0.5*L_dice +    │
        │             0.5*L_bce       │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │   BACKPROPAGATION & OPTIMIZE    │
        │   • Gradient computation        │
        │   • Gradient clipping (norm≤1.0)│
        │   • Adam optimizer update       │
        │   • Learning rate scheduler     │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │   EVALUATION (training/)        │
        │   Compute metrics per batch:    │
        │   • Dice Similarity (DSC)       │
        │   • Intersection over Union     │
        │   • F1-Score                    │
        │   • Per-class metrics           │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │   CHECKPOINT SAVING             │
        │   Save every N epochs           │
        │   Keep best validation model    │
        │   Save final model              │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │   INFERENCE (inference/)        │
        │   Load best model checkpoint    │
        │   Forward pass (no_grad)        │
        │   Argmax for class prediction   │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │   OUTPUT: Segmentation Mask     │
        │   (240×240×155 with 4 classes)  │
        │   Save as NIfTI file            │
        └──────────────────────────────────┘
```

---

## 📌 EACH COMPONENT'S RESPONSIBILITY

### 1. DATA PIPELINE (data/)

**Files:**

- `preprocessing.py` → NIfTI loading, normalization, resizing
- `augmentation.py` → Spatial & intensity transformations
- `dataloader.py` → PyTorch DataLoader with lazy loading

**Input:**

- Raw BraTS NIfTI files (T1, T1ce, T2, FLAIR modalities)
- Manual segmentation ground truth (label.nii.gz)

**Output:**

- Normalized 4D tensors: (4 channels, 240, 240, 155)
- Segmentation labels: (1, 240, 240, 155) with values 0-3

**Key Functions:**

```python
NIfTIPreprocessor.load_nifti()           # Load .nii.gz
NIfTIPreprocessor.normalize()            # Z-score normalization
NIfTIPreprocessor.resize()               # Standardize to fixed size
DataAugmentor.apply_augmentation()       # Random transforms
BraTS2021DataLoader.__getitem__()        # Batch creation
```

---

### 2. MODEL ARCHITECTURE (models/)

**unet_attention.py:**

- `AttentionUNet3D` → Full model
- `EncoderBlock` → Downsampling (4 levels)
- `DecoderBlock` → Upsampling with attention
- `BottleneckBlock` → Feature extraction

**attention_gates.py:**

- `AttentionGate` → Spatial attention (U-Net style)
- `ChannelAttention` → Channel-wise attention (SE-Net style)
- `DoubleConvBlock3D` → Conv→BN→ReLU→Conv→BN→ReLU
- `ConvBlock3D` → Single convolution block

**loss_functions.py:**

- `DiceLoss` → Dice = 2|X∩Y|/(|X|+|Y|)
- `BCELoss` → Binary cross-entropy
- `DiceBCELoss` → Weighted combination (0.5 each)

**Input:**

- 4D tensor: (Batch, 4 channels, 240, 240, 155)

**Output:**

- 4D tensor: (Batch, 4 classes, 240, 240, 155) [logits]

**Key Methods:**

```python
AttentionUNet3D.forward(x)    # Forward pass
DiceBCELoss.forward(pred, target)  # Loss computation
```

---

### 3. TRAINING PIPELINE (training/)

**train.py:**

- `Trainer` class manages entire training

**metrics.py:**

- Dice, IoU, F1-Score calculation
- Per-class & mean metrics
- MetricAggregator for batch averaging

**Responsibilities:**

1. Load data (train/val loaders)
2. Initialize model, optimizer, scheduler
3. Loop for N epochs:
   - Forward pass on batches
   - Compute loss
   - Backpropagation
   - Gradient clipping
   - Update weights
   - Compute metrics
4. Validation every N batches
5. Learning rate scheduling
6. Early stopping
7. Checkpoint saving

**Key Methods:**

```python
Trainer.train_epoch()          # One training epoch
Trainer.validate()             # Validation loop
Trainer.train()                # Full training
```

---

### 4. INFERENCE PIPELINE (inference/)

**predict.py:**

- `ModelPredictor` class for inference

**postprocess.py:**

- `PostProcessor` for output conversion

**Workflow:**

1. Load model weights from checkpoint
2. Load unseen MRI volume (NIfTI)
3. Preprocess (normalize, resize)
4. Forward pass through model
5. Argmax to get class predictions
6. Apply threshold if needed
7. Post-process (morphological ops)
8. Save as NIfTI file

**Key Methods:**

```python
ModelPredictor.predict(mri_path)  # Single prediction
ModelPredictor.predict_batch()    # Batch predictions
PostProcessor.save_nifti()        # Save output
```

---

### 5. REST API (api/)

**main.py:**

- FastAPI app initialization
- CORS setup
- Error handling middleware

**routes.py:**

- `POST /api/predict` → Upload MRI → Get segmentation
- `GET /api/model-info` → Model metadata
- `GET /api/health` → Health check

**schemas.py:**

- Request models (file upload)
- Response models (JSON results)

**utils.py:**

- Helper functions
- File I/O
- Model loading

**Workflow:**

```
Client sends MRI file
    ↓
API receives & saves file
    ↓
ModelPredictor.predict(file)
    ↓
Save result to outputs/
    ↓
Return download URL to client
```

---

### 6. WEB UI (ui/)

**Components:**

- `ImageUploader` → Drag-drop file upload
- `MRIViewer` → Interactive 3D slice navigation
- `ResultsDisplay` → Segmentation visualization
- `MetricsDisplay` → Performance metrics

**Functionality:**

1. User uploads NIfTI file
2. Send to API via POST /api/predict
3. Receive segmentation result
4. Display overlays on original slices
5. Allow download of result

---

### 7. EXPERIMENT & ABLATION (experiments/)

**baseline_unet.py:**

- Standard U-Net without attention
- For comparison with Attention U-Net

**analysis.py:**

- Results visualization
- Metric plotting
- Comparison charts

**Purpose:** Research & validation of attention mechanisms

---

### 8. TESTS (tests/)

**test_models.py:**

- Model forward pass
- Output shape verification
- Gradient computation

**test_preprocessing.py:**

- Data loading
- Normalization
- Resizing

**test_dataloader.py:**

- DataLoader iteration
- Batch shape verification
- Augmentation application

**Purpose:** Ensure quality & reproducibility

---

## 📥 INPUT SPECIFICATIONS

### Training Data Input

**Format:** BraTS 2021 dataset

- **File Type:** NIfTI (.nii.gz)
- **Modalities:** 4 channels (T1, T1ce, T2, FLAIR)
- **Original Size:** Variable (typically ~240×240×155)
- **Standardized Size:** 240×240×155 after preprocessing
- **Data Type:** float32

**Structure:**

```
BraTS_dataset/
├── HGG/                           # High-grade glioma (training)
│   ├── BraTS_001/
│   │   ├── BraTS_001_t1.nii.gz
│   │   ├── BraTS_001_t1ce.nii.gz
│   │   ├── BraTS_001_t2.nii.gz
│   │   ├── BraTS_001_flair.nii.gz
│   │   └── BraTS_001_seg.nii.gz     # Ground truth segmentation
│   └── ... (up to 259 cases)
│
└── LGG/                           # Low-grade glioma (training)
    ├── BraTS_101/
    │   └── ... (same structure)
    └── ... (76 cases)
```

**Total:** 335 cases (259 HGG + 76 LGG)

**Split:**

- Training: 268 cases (80%)
- Validation: 34 cases (10%)
- Testing: 33 cases (10%)

---

## 📤 OUTPUT SPECIFICATIONS

### Model Output (Training)

**Saved Checkpoints:**

```python
{
    "epoch": 50,
    "model_state_dict": {...},       # Model weights
    "optimizer_state_dict": {...},   # Optimizer state
    "best_dice": 0.85,
    "train_loss": 0.15,
    "val_loss": 0.18,
    "metrics": {
        "dice": 0.85,
        "iou": 0.75,
        "f1": 0.82
    }
}
```

**Location:** `checkpoints/best_model.pth` (best validation)

### Inference Output (Prediction)

**Format:** NIfTI file (.nii.gz)

- **Shape:** 240×240×155 (same as input)
- **Data Type:** uint8
- **Values:** 0-3 (class indices)
  - 0 = Background
  - 1 = Necrotic Core
  - 2 = Peritumoral Edema
  - 3 = Enhancing Tumor

**Saved at:** `outputs/predictions/{patient_id}_seg.nii.gz`

**JSON Metadata (API):**

```json
{
	"patient_id": "BraTS_001",
	"prediction_time": "0.45s",
	"file_url": "/api/download/BraTS_001_seg.nii.gz",
	"metrics": {
		"processing_time_ms": 450,
		"file_size_mb": 2.1
	}
}
```

---

## 📊 EVALUATION METRICS

All computed in `training/metrics.py`:

### 1. Dice Similarity Coefficient (DSC)

```
DSC = 2|X∩Y| / (|X| + |Y|)
Range: [0, 1]
Better = Higher
Use Case: Primary metric for medical image segmentation
```

### 2. Intersection over Union (IoU / Jaccard Index)

```
IoU = |X∩Y| / |X∪Y|
Range: [0, 1]
Better = Higher
Use Case: Standard computer vision metric
```

### 3. F1-Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Range: [0, 1]
Better = Higher
Use Case: Balance between precision and recall
```

### 4. Per-Class Metrics

- Computed separately for each of 4 classes
- Background usually excluded from mean
- Important for understanding class-specific performance

**Expected Performance (Preliminary):**
| Model | DSC Mean | IoU Mean | F1-Score |
|-------|----------|----------|----------|
| Standard U-Net | 0.88 | 0.79 | 0.87 |
| Attention U-Net | 0.91 | 0.84 | 0.90 |
| Target (to achieve) | >0.90 | >0.83 | >0.89 |

---

## 🔄 EXECUTION FLOW

### TRAINING MODE

```
1. Load config.py settings
2. Download/Load BraTS dataset (335 cases)
3. Preprocess: Normalize, resize to 240×240×155
4. Create DataLoader (batch_size=16, workers=4)
5. Initialize AttentionUNet3D model
6. Initialize DiceBCELoss + Adam optimizer
7. Setup ReduceLROnPlateau scheduler
8. Setup TensorBoard logging

Loop for NUM_EPOCHS (100):
  ├─ Training Phase:
  │  └─ For each batch in train_loader:
  │     ├─ Forward pass: y_pred = model(x)
  │     ├─ Compute loss: L = loss_fn(y_pred, y_true)
  │     ├─ Backward pass: L.backward()
  │     ├─ Clip gradients: clip_grad_norm_(norm=1.0)
  │     ├─ Optimizer step: optimizer.step()
  │     ├─ Compute metrics: DSC, IoU, F1
  │     └─ Log to TensorBoard
  │
  ├─ Validation Phase:
  │  ├─ Disable gradients: with torch.no_grad()
  │  ├─ For each batch in val_loader:
  │  │  ├─ Forward pass only (no backprop)
  │  │  ├─ Compute loss & metrics
  │  │  └─ Aggregate results
  │  └─ Check early stopping condition
  │
  ├─ Learning Rate Scheduling:
  │  └─ Reduce LR if val_metric plateaus
  │
  └─ Checkpoint Saving (every 5 epochs):
     └─ Save if validation DSC > best_seen

9. Save final_model.pth after training
10. Generate training report (metrics, times, etc.)
```

**Total Training Time:** ~15-18 hours on 2× T4 GPUs (335 cases, 50 epochs)

---

### INFERENCE MODE

```
1. Load best_model.pth weights
2. Receive new MRI file (NIfTI)
3. Preprocess:
   ├─ Load with nibabel
   ├─ Z-score normalize
   ├─ Resize to 240×240×155
   └─ Create tensor (1, 4, 240, 240, 155)
4. Forward pass (no_grad):
   └─ pred_logits = model(input_tensor)
5. Post-process:
   ├─ Argmax → get class indices
   ├─ Confidence threshold (optional)
   ├─ Morphological operations (optional)
   └─ Convert back to NIfTI
6. Save result as NIfTI file
7. Return to API or user
```

**Inference Time:** ~0.5-2 seconds per volume (depending on hardware)

---

### API DEPLOYMENT

```
1. Start FastAPI server:
   uvicorn api.main:app --host 0.0.0.0 --port 8000

2. Load model on startup:
   └─ best_model.pth loaded to GPU/CPU

3. Wait for requests:
   ├─ POST /api/predict
   │  ├─ Receive file
   │  ├─ Run inference (above)
   │  ├─ Save result
   │  └─ Return JSON with URL
   │
   ├─ GET /api/model-info
   │  └─ Return model metadata
   │
   └─ GET /api/health
      └─ Return status 200

4. Serve React UI on port 3000:
   npm start
```

---

## 🔑 KEY DESIGN DECISIONS

1. **Architecture:** Attention-Enhanced U-Net
   - Why? Better tumor focus than standard U-Net
   - Proven on medical imaging tasks

2. **Loss Function:** Dice-BCE (50%-50%)
   - Why? Dice handles class imbalance, BCE provides stability

3. **Batch Size:** 16
   - Why? Balance between memory & gradient stability

4. **Learning Rate:** 1e-3 with ReduceLROnPlateau
   - Why? Conservative start with adaptive reduction

5. **Data Augmentation:** Heavy (rotation, flip, elastic, intensity)
   - Why? Small dataset (335 cases) needs augmentation

6. **Lazy Loading:** Load each volume on-demand
   - Why? All 335 cases don't fit in RAM (~2GB each)

7. **API + UI:** Separate concerns
   - API: FastAPI (Python)
   - UI: React (TypeScript)
   - Why? Modular, scalable, standard architecture

---

## 🚀 DEPLOYMENT CHECKLIST

- [ ] Train model to convergence (>85% DSC)
- [ ] Evaluate on test set
- [ ] Save best_model.pth
- [ ] Test API locally
- [ ] Test UI locally
- [ ] Build Docker image
- [ ] Deploy to cloud (GCP/AWS)
- [ ] Monitor inference performance
- [ ] Collect user feedback

---

## 📋 EXPECTED EXECUTION TIME

| Phase                      | Time          | Hardware      |
| -------------------------- | ------------- | ------------- |
| Data download              | 30 min        | CPU + Network |
| Data preprocessing         | 1 hour        | CPU           |
| Model training (50 epochs) | 15-18 hours   | 2× T4 GPU     |
| Model evaluation           | 30 min        | 1× T4 GPU     |
| API testing                | 10 min        | CPU           |
| **Total**                  | **~25 hours** |               |

---

## 🔗 KEY FILES TO UNDERSTAND

**Start here:**

1. `README.md` - Project overview
2. `config.py` - All settings centralized
3. `models/unet_attention.py` - Model architecture
4. `training/train.py` - Training loop
5. `api/main.py` - API endpoints
6. `RUN_GUIDE.md` - Step-by-step execution

---

## 📞 TROUBLESHOOTING COMMON ISSUES

### Memory Error During Training

→ Reduce BATCH_SIZE in config.py (try 8 or 4)

### Model Not Converging

→ Check learning rate, increase epochs, verify data loading

### API Port Already in Use

→ Change API_PORT in config.py or kill process: `lsof -i :8000`

### Data Download Fails

→ Verify Kaggle credentials, check internet connection

### GPU Not Detected

→ Check CUDA installation, verify PyTorch compiled with CUDA

---

**Version:** 1.0.0-alpha
**Last Updated:** April 2026
**Status:** Active Development
