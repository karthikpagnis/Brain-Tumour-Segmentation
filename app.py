"""
Brain Tumor Segmentation - FastAPI Server
REST API + Web Interface for inference
"""

import os
import io
import json
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Brain Tumor Segmentation",
    description="Attention-Enhanced U-Net for MRI Brain Tumor Segmentation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "best_model.pth"  # Place this file in root directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Model config (must match training config)
IN_CHANNELS = 4
NUM_CLASSES = 4
BASE_FILTERS = 32

LABEL_NAMES = {
    0: "Background",
    1: "Necrotic Core",
    2: "Peritumoral Edema",
    3: "Enhancing Tumor"
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (from your training)
# ─────────────────────────────────────────────────────────────────────────────

import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(g_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(x_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base_filters=32, dropout=0.1):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f*2, dropout)
        self.enc3 = DoubleConv(f*2, f*4, dropout)
        self.enc4 = DoubleConv(f*4, f*8, dropout)
        self.bottleneck = DoubleConv(f*8, f*16, dropout)
        self.pool = nn.MaxPool3d(2)
        self.up4 = nn.ConvTranspose3d(f*16, f*8, 2, stride=2)
        self.up3 = nn.ConvTranspose3d(f*8, f*4, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(f*4, f*2, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(f*2, f, 2, stride=2)
        self.att4 = AttentionGate(f*8, f*8, f*4)
        self.att3 = AttentionGate(f*4, f*4, f*2)
        self.att2 = AttentionGate(f*2, f*2, f)
        self.att1 = AttentionGate(f, f, f//2)
        self.dec4 = DoubleConv(f*16, f*8)
        self.dec3 = DoubleConv(f*8, f*4)
        self.dec2 = DoubleConv(f*4, f*2)
        self.dec1 = DoubleConv(f*2, f)
        self.out_conv = nn.Conv3d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.out_conv(d1)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

try:
    model = AttentionUNet3D(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_filters=BASE_FILTERS,
        dropout=0.0
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_nifti(file_bytes):
    """Load NIfTI file from bytes - handles both .nii and .nii.gz"""
    try:
        import tempfile
        # Write to temp file for more robust loading
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            nifti_img = nib.load(tmp_path)
            data = nifti_img.get_fdata(dtype=np.float32)
            affine = nifti_img.affine
            return data, affine
        finally:
            # Clean up temp file
            os.remove(tmp_path)
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI: {str(e)}")

def preprocess_volume(volume):
    """Normalize 3D volume - handles both channel orders (C,H,W,D) or (H,W,D,C)"""
    # Handle different channel orders
    if volume.ndim == 4:
        # If shape is (H, W, D, C), transpose to (C, H, W, D)
        if volume.shape[-1] == 4:
            print(f"Converting channels-last {volume.shape} to channels-first")
            volume = np.transpose(volume, (3, 0, 1, 2))
        # If shape is (C, H, W, D), keep as is
        elif volume.shape[0] == 4:
            print(f"Volume already in channels-first format: {volume.shape}")
        else:
            raise ValueError(f"Expected 4 channels, got shape {volume.shape}")
    elif volume.ndim == 3:
        print(f"Adding channel dimension to 3D volume: {volume.shape}")
        volume = volume[np.newaxis]  # Add channel dim
    else:
        raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D with shape {volume.shape}")

    normalized = np.zeros_like(volume, dtype=np.float32)
    for c in range(volume.shape[0]):
        mask = volume[c] > 0
        if mask.sum() > 0:
            mu = volume[c][mask].mean()
            std = volume[c][mask].std() + 1e-8
            normalized[c] = (volume[c] - mu) / std
        else:
            normalized[c] = volume[c]

    return normalized

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(volume_np):
    """Run inference on 4-channel volume"""
    tensor = torch.from_numpy(volume_np).unsqueeze(0).to(DEVICE)  # (1,4,H,W,D)

    with torch.cuda.amp.autocast(enabled=True):
        logits = model(tensor)

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred[pred == 3] = 4  # Map class 3 → BraTS label 4

    return pred

# ─────────────────────────────────────────────────────────────────────────────
# REST API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web interface"""
    return get_html_interface()

@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/info")
async def model_info():
    """Model information"""
    return {
        "name": "Attention-Enhanced U-Net 3D",
        "input_channels": IN_CHANNELS,
        "output_classes": NUM_CLASSES,
        "class_names": LABEL_NAMES,
        "best_dice": 0.6189,
        "best_epoch": 30,
        "training_data": "BraTS 2019 (335 cases)",
        "inference_time_ms": "~1000-2000 ms",
    }

@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict segmentation for uploaded NIfTI file

    Returns:
    - segmentation: predicted mask (4 classes)
    - metrics: per-class statistics
    - file_url: download link
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if not file.filename.endswith(('.nii', '.nii.gz', '.nifti')):
            raise HTTPException(status_code=400, detail="Only NIfTI files allowed (.nii, .nii.gz)")

        # Read file
        contents = await file.read()
        volume, affine = load_nifti(contents)

        # Check for valid 4-channel volume (handles both channel orders)
        is_valid = False
        if volume.ndim == 4:
            if volume.shape[0] == 4 or volume.shape[-1] == 4:
                is_valid = True

        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Expected 4-channel NIfTI volume, got shape {volume.shape}")

        # Preprocess
        normalized = preprocess_volume(volume)

        # Inference
        logger.info(f"Running inference on {file.filename}...")
        prediction = predict(normalized)

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"pred_{timestamp}.nii.gz"
        output_path = UPLOAD_DIR / output_filename

        nib.save(nib.Nifti1Image(prediction, affine), str(output_path))

        # Compute metrics
        metrics = {}
        for cls in range(1, NUM_CLASSES):
            voxels = (prediction == cls).sum()
            metrics[LABEL_NAMES[cls]] = int(voxels)

        logger.info(f"✅ Prediction saved: {output_path}")

        return {
            "status": "success",
            "filename": output_filename,
            "shape": list(prediction.shape),
            "classes_found": {LABEL_NAMES[i]: int((prediction == i).sum()) for i in range(NUM_CLASSES)},
            "download_url": f"/download/{output_filename}",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download prediction file"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename, media_type="application/gzip")

# ─────────────────────────────────────────────────────────────────────────────
# WEB INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def get_html_interface():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Segmentation</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
                font-size: 28px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #f8f9ff;
            }
            .upload-area:hover {
                border-color: #764ba2;
                background: #f0f2ff;
            }
            .upload-area.dragover {
                border-color: #764ba2;
                background: #e8eaff;
            }
            .upload-icon {
                font-size: 48px;
                margin-bottom: 15px;
            }
            .upload-text {
                color: #333;
                font-weight: 600;
                margin-bottom: 5px;
            }
            .upload-hint {
                color: #999;
                font-size: 12px;
            }
            input[type="file"] { display: none; }
            .button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 20px;
                transition: all 0.3s ease;
                width: 100%;
            }
            .button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            .button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .results {
                margin-top: 30px;
                padding: 20px;
                background: #f0f2ff;
                border-radius: 10px;
                display: none;
            }
            .results.show { display: block; }
            .result-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                border-left: 4px solid #667eea;
            }
            .result-label {
                font-weight: 600;
                color: #333;
                margin-bottom: 5px;
            }
            .result-value {
                color: #666;
                font-size: 14px;
            }
            .download-btn {
                background: #27ae60;
                margin-top: 15px;
                display: inline-block;
                text-decoration: none;
                text-align: center;
            }
            .download-btn:hover {
                background: #229954;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                background: #fee;
                color: #c33;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                display: none;
                border-left: 4px solid #c33;
            }
            .error.show { display: block; }
            .model-info {
                background: #e8f4f8;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 13px;
                color: #333;
                border-left: 4px solid #3498db;
            }
            .model-info strong { color: #2c3e50; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Brain Tumor Segmentation</h1>
            <p class="subtitle">Attention-Enhanced U-Net for MRI Analysis</p>

            <div class="model-info">
                <strong>Model:</strong> Attention-Enhanced U-Net 3D<br>
                <strong>Best Dice:</strong> 0.6189 (30 epochs)<br>
                <strong>Dataset:</strong> BraTS 2019 (335 cases)<br>
                <strong>Classes:</strong> Necrotic Core, Edema, Enhancing Tumor
            </div>

            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Drop your MRI file here</div>
                <div class="upload-hint">or click to browse (.nii, .nii.gz)</div>
                <input type="file" id="fileInput" accept="*">
            </div>

            <button class="button" id="submitBtn">Upload & Predict</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your MRI file...</p>
            </div>

            <div class="error" id="error"></div>

            <div class="results" id="results">
                <div class="result-item">
                    <div class="result-label">Prediction Status</div>
                    <div class="result-value" id="status">✅ Segmentation complete!</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Output File</div>
                    <div class="result-value" id="filename"></div>
                </div>
                <div class="result-item">
                    <div class="result-label">Classes Detected</div>
                    <div class="result-value" id="classes"></div>
                </div>
                <button class="button download-btn" id="downloadBtn">⬇️ Download Prediction</button>
            </div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const results = document.getElementById('results');
            const downloadBtn = document.getElementById('downloadBtn');

            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                fileInput.files = e.dataTransfer.files;
            });

            submitBtn.addEventListener('click', async () => {
                if (!fileInput.files.length) {
                    showError('Please select a file');
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                loading.style.display = 'block';
                error.classList.remove('show');
                results.classList.remove('show');

                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Prediction failed');
                    }

                    document.getElementById('filename').textContent = data.filename;
                    const classesHtml = Object.entries(data.classes_found)
                        .map(([name, count]) => `${name}: ${count.toLocaleString()} voxels`)
                        .join('<br>');
                    document.getElementById('classes').innerHTML = classesHtml;

                    downloadBtn.onclick = () => window.location.href = data.download_url;
                    results.classList.add('show');

                } catch (err) {
                    showError(err.message);
                } finally {
                    loading.style.display = 'none';
                }
            });

            function showError(msg) {
                error.textContent = '❌ ' + msg;
                error.classList.add('show');
            }
        </script>
    </body>
    </html>
    """

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Brain Tumor Segmentation API starting...")
    if model is not None:
        logger.info("✅ Model ready for inference")
    else:
        logger.warning("⚠️ Model not loaded - check best_model.pth")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
