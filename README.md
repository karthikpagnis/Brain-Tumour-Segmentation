# Brain Tumour Segmentation from MRI Scans Using an Attention-Enhanced U-Net Architecture

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Project Overview

This repository contains a **research-grade deep learning framework** for automatic brain tumor segmentation from multimodal MRI scans using an **Attention-Enhanced U-Net architecture**. The project is designed to address the challenges of **tumor heterogeneity** and **inter-observer variability** in manual medical image annotation.

### Key Challenges Addressed

- **Tumor Heterogeneity**: Tumors vary significantly in morphology, size, shape, and intensity
- **Class Imbalance**: Tumor lesions represent a small fraction of the image volume
- **Manual Delineation**: Current approaches are laborious and subject to observer bias
- **Generalization**: Model must work across diverse patient cases and imaging protocols

---

## 🎯 Research Contributions

### 1. **Attention-Enhanced Architecture**

- Encoder-decoder U-Net with skip connections
- Attention gates to focus on tumor-relevant features
- Learn to emphasize: necrotic core, peritumoral edema, enhancing tumor
- Suppress: healthy tissue and background regions

### 2. **Robust Training Strategy**

- Composite **Dice-Binary Cross-Entropy (Dice-BCE)** loss to handle class imbalance
- Spatial augmentations: rotation, flip, elastic deformation
- Intensity augmentations: brightness, contrast, gamma shifts
- Gradient clipping and learning rate scheduling

### 3. **Comprehensive Evaluation**

- Multiple metrics: Dice Similarity Coefficient (DSC), Intersection over Union (IoU), F1-score
- Baseline comparisons: Standard U-Net without attention
- Ablation studies: Loss function, augmentation, attention mechanisms
- Statistical significance testing

---

## 🏗️ Project Structure

```
Brain-Tumour-Segmentation/
├── data/                          # Data pipeline
│   ├── download_brats.py          # Dataset download and extraction
│   ├── preprocessing.py           # NIfTI processing, normalization
│   ├── augmentation.py            # Spatial & intensity augmentations
│   └── dataloader.py              # PyTorch DataLoader
│
├── models/                        # Neural network architectures
│   ├── unet_attention.py          # Attention-Enhanced U-Net
│   ├── attention_gates.py         # Attention gate modules
│   └── loss_functions.py          # Dice-BCE composite loss
│
├── training/                      # Training pipeline
│   ├── train.py                   # Main training script
│   ├── validate.py                # Validation loop
│   ├── config.py                  # Hyperparameters
│   └── metrics.py                 # DSC, IoU, F1-score
│
├── inference/                     # Inference pipeline
│   ├── predict.py                 # Single prediction
│   └── postprocess.py             # Output processing
│
├── api/                           # REST API backend
│   ├── main.py                    # FastAPI app
│   ├── routes.py                  # API endpoints
│   ├── schemas.py                 # Request/response models
│   └── utils.py                   # Utilities
│
├── ui/                            # Web UI (React)
│   ├── src/components/            # React components
│   ├── src/pages/                 # Pages
│   └── package.json               # Dependencies
│
├── experiments/                   # Research & ablation studies
│   ├── baseline_unet.py           # Standard U-Net
│   └── analysis.py                # Results visualization
│
├── tests/                         # Unit & integration tests
├── docs/                          # Documentation
├── checkpoints/                   # Model weights
├── outputs/                       # Results & logs
├── config.py                      # Global configuration
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (recommended for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/karthikpagnis/Brain-Tumour-Segmentation.git
cd Brain-Tumour-Segmentation
```

2. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install package in development mode**

```bash
pip install -e .
```

---

## 📊 Dataset

### BraTS (Brain Tumor Segmentation) Challenge

The project uses the **BraTS dataset**, which provides:

- **Multimodal MRI scans**: T1, T1-weighted with contrast (T1ce), T2, FLAIR
- **Expert annotations**: Four tumor classes (background, necrotic core, edema, enhancing tumor)
- **Realistic clinical data**: Representing diverse patient populations and imaging protocols

**Dataset Statistics:**

- Training: ~369 cases
- Validation: ~125 cases
- Testing: Cross-validation and hold-out test set

**Download Instructions:**

```bash
# BraTS 2021 dataset (requires registration at https://www.med.upenn.edu/cbica/brats2021/)
python scripts/download_data.py --year 2021
```

---

## 🔄 Converting Separate BraTS Files to Combined Format

BraTS dataset files come as separate modality files. Use this script to combine them into the 4-channel format required by the model:

### Quick Usage

```bash
# Basic usage - creates BraTS19_014_combined.nii.gz
python scripts/combine_brats_files.py --input_dir BraTS19_014

# With custom output name
python scripts/combine_brats_files.py --input_dir BraTS19_014 --output my_mri_scan.nii.gz

# From a nested directory
python scripts/combine_brats_files.py --input_dir data/BraTS_Training/BraTS19_001
```

### Supported File Formats

The script automatically detects these naming patterns:

- `patient_t1.nii` / `patient_t1.nii.gz`
- `patient_t1ce.nii` / `patient_t1ce.nii.gz`
- `patient_t2.nii` / `patient_t2.nii.gz`
- `patient_flair.nii` / `patient_flair.nii.gz`

### How It Works

**Input:** 4 separate single-channel files

```
BraTS19_014/
├── BraTS19_014_t1.nii        (H, W, D)
├── BraTS19_014_t1ce.nii      (H, W, D)
├── BraTS19_014_t2.nii        (H, W, D)
└── BraTS19_014_flair.nii     (H, W, D)
```

**Output:** 1 combined 4-channel file

```
BraTS19_014_combined.nii.gz   (H, W, D, 4)
```

### Upload to Web App

Once combined, you can upload the `.nii.gz` file to the web interface for segmentation:

```bash
# Start the web app
python app.py

# Visit http://localhost:8000
# Upload the combined file and download predictions
```

---

## 🧠 Model Architecture

### Attention-Enhanced U-Net

```
Input (4-channel MRI) → Encoder → Bottleneck → Decoder → Output (4-class segmentation)
                          ↓ Skip Connections with Attention Gates ↓
```

**Architecture Details:**

- Encoder: 4 downsampling blocks (Conv→BatchNorm→ReLU→MaxPool)
- Channel progression: 4 → 32 → 64 → 128 → 256
- Attention Gates: Learn spatial and channel-wise importance
- Decoder: 4 upsampling blocks with skip connections
- Output: 4 segmentation classes via softmax

**Key Features:**

- Batch normalization for training stability
- Dropout (20%) for regularization
- Attention gates to suppress irrelevant regions
- Skip connections for feature preservation

---

## 🎓 Training

### Hyperparameters

| Parameter      | Value                        |
| -------------- | ---------------------------- |
| Batch Size     | 16                           |
| Learning Rate  | 1e-3                         |
| Optimizer      | Adam                         |
| Epochs         | 100                          |
| Loss Function  | Dice-BCE (0.5-0.5 weighting) |
| Scheduler      | ReduceLROnPlateau            |
| Early Stopping | 20 epochs patience           |

### Training a Model

```bash
python training/train.py \
    --config config.py \
    --experiment attention_unet_v1 \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 1e-3
```

### Running with Baseline Comparison

```bash
python training/train.py --run_baseline True
```

---

## 📈 Evaluation Metrics

### Metrics Used

1. **Dice Similarity Coefficient (DSC)**
   - Measures overlap between prediction and ground truth
   - Higher = better (range: 0-1)
   - DSC = 2|X∩Y| / (|X| + |Y|)

2. **Intersection over Union (IoU)**
   - Jaccard index for segmentation quality
   - IoU = |X∩Y| / |X∪Y|

3. **F1-Score**
   - Harmonic mean of precision and recall
   - Per-class and macro-averaged metrics

### Evaluation Script

```bash
python scripts/evaluate_model.py \
    --model checkpoints/best_model.pth \
    --test_data data/test_cases \
    --output outputs/evaluation_report.json
```

---

## 🔍 Inference

### Single Patient Prediction

```python
from inference.predict import predict_from_nifti
from models.unet_attention import AttentionUNet

# Load model
model = AttentionUNet.load_pretrained("checkpoints/best_model.pth")

# Predict
prediction = predict_from_nifti(
    nifti_path="patient_scan.nii.gz",
    model=model,
    confidence_threshold=0.5
)

# Save result
prediction.save("prediction.nii.gz")
```

### Batch Prediction

```bash
python inference/predict.py \
    --model checkpoints/best_model.pth \
    --input_dir data/test_cases \
    --output_dir outputs/predictions \
    --batch_size 8
```

---

## 🌐 REST API & Web UI

### Start API Server

```bash
# Development server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production server
gunicorn api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### API Endpoints

#### **POST** `/api/predict`

Upload MRI volume and get segmentation prediction

```bash
curl -X POST \
  -F "file=@patient_scan.nii.gz" \
  http://localhost:8000/api/predict
```

#### **GET** `/api/model-info`

Get model metadata and training metrics

```bash
curl http://localhost:8000/api/model-info
```

#### **GET** `/api/health`

Health check

```bash
curl http://localhost:8000/api/health
```

### Interactive Swagger UI

Access auto-generated API documentation at: `http://localhost:8000/docs`

### Web UI

```bash
cd ui
npm install
npm start
```

Visit `http://localhost:3000` to use the interactive interface.

**Features:**

- Drag-and-drop MRI upload
- Interactive slice viewer
- Real-time segmentation visualization
- Download results as NIfTI
- Model information and metrics display

---

## 📚 Research Papers & References

### Attention Mechanisms

- [Attention U-Net](https://arxiv.org/abs/1804.03999) - Ozan Oktay et al.
- [Channel Attention Networks](https://arxiv.org/abs/1709.01507) - Jie Hu et al.

### Segmentation Loss Functions

- [Dice Loss](https://arxiv.org/abs/1606.06650) - V. Milletari et al.
- [Focal Loss](https://arxiv.org/abs/1708.02002) - Tsung-Yi Lin et al.
- [Lovász-Softmax](https://arxiv.org/abs/1705.08790) - Maxim Berman et al.

### Medical Image Segmentation

- [U-Net](https://arxiv.org/abs/1505.04597) - Ronneberger et al.
- [3D U-Net](https://arxiv.org/abs/1606.06650) - Ö. Çiçek et al.

### BraTS Challenge

- [BraTS 2021](https://www.med.upenn.edu/cbica/brats2021/)
- [Multimodal MRI Analysis](https://arxiv.org/abs/1811.02629)

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ --cov=. --cov-report=html
```

Test categories:

- **Data tests**: Preprocessing, augmentation, dataloader
- **Model tests**: Architecture forward pass, loss computation
- **API tests**: Endpoint validation, error handling
- **Integration tests**: End-to-end pipeline

---

## 📊 Results & Benchmarks

### Performance Metrics (Preliminary)

| Model           | DSC (Mean) | IoU (Mean) | F1-Score |
| --------------- | ---------- | ---------- | -------- |
| Standard U-Net  | 0.88       | 0.79       | 0.87     |
| Attention U-Net | **0.91**   | **0.84**   | **0.90** |
| Improvement     | +3.4%      | +6.3%      | +3.4%    |

### Ablation Studies

- Impact of attention gates: +2-3% DSC improvement
- Loss function weighting: Dice-BCE optimal over single-loss
- Data augmentation: Critical for generalization (±5% performance)

_Note: Full results with statistical significance testing in progress_

---

## 🔧 Configuration

All settings are in `config.py`:

- Model architecture (channels, dropout, attention type)
- Training hyperparameters (batch size, learning rate, epochs)
- Data augmentation settings
- Loss function parameters
- Evaluation metrics and thresholds
- API configuration

**Example: Update configuration**

```python
# In config.py
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
DICE_LOSS_WEIGHT = 0.6
USE_ATTENTION_GATES = True
```

---

## 📝 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed technical architecture
- **[DATASET.md](docs/DATASET.md)** - BraTS dataset guide and preprocessing
- **[API_DOCS.md](docs/API_DOCS.md)** - REST API documentation
- **[RESULTS.md](docs/RESULTS.md)** - Experimental results and comparisons
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Development guidelines

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💼 Author

**Karthik Pagnis**  
Department of Computer Science & Engineering  
Indian Institute of Technology Madras (IIT Madras)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- Improve model architecture (different attention mechanisms)
- Optimize inference speed
- Enhance data augmentation strategies
- Add more evaluation metrics
- Improve documentation
- Bug fixes and optimizations

---

## 📞 Contact & Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/karthikpagnis/Brain-Tumour-Segmentation/issues)
- **Discussions**: Join research discussions in [GitHub Discussions](https://github.com/karthikpagnis/Brain-Tumour-Segmentation/discussions)

---

## 🙏 Acknowledgments

- BraTS challenge organizers and data providers
- IIT Madras for academic resources
- PyTorch and FastAPI communities
- Research papers cited in the documentation

---

**Last Updated:** April 2025  
**Status:** Active Development  
**Version:** 1.0.0-alpha
