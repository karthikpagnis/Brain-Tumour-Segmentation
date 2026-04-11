"""
FastAPI Backend for Brain Tumor Segmentation
REST API for model inference with automatic Swagger UI documentation
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import nibabel as nib

from config import (
    API_HOST,
    API_PORT,
    ALLOWED_ORIGINS,
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    DEVICE,
    NUM_CLASSES,
)
from models.unet_attention import AttentionUNet3D
from data.preprocessing import NIfTIPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="REST API for automatic brain tumor segmentation from MRI scans",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and device
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
model = None
preprocessor = None


# Pydantic models for API schema
class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""

    status: str
    message: str
    processing_time: float
    output_filename: Optional[str] = None
    predictions_url: Optional[str] = None
    class_names: dict = {
        "0": "Background",
        "1": "Necrotic Core",
        "2": "Peritumoral Edema",
        "3": "Enhancing Tumor",
    }


class HealthResponse(BaseModel):
    """Response schema for health check"""

    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""

    model_name: str
    model_type: str
    input_channels: int
    output_classes: int
    parameters: int
    device: str
    attention_gates_enabled: bool


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, preprocessor

    try:
        logger.info("Loading model on startup...")

        # Try to load best model
        best_model_path = CHECKPOINTS_DIR / "attention_unet_v1_best.pth"

        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model = AttentionUNet3D().to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {best_model_path}")
        else:
            # Create new model if no checkpoint exists
            model = AttentionUNet3D().to(device)
            logger.info("Created new model (no checkpoint found)")

        model.eval()
        preprocessor = NIfTIPreprocessor()

        logger.info(f"Model loaded successfully on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(p.numel() for p in model.parameters())

    return ModelInfoResponse(
        model_name="Attention-Enhanced U-Net",
        model_type="3D Encoder-Decoder",
        input_channels=4,
        output_classes=NUM_CLASSES,
        parameters=total_params,
        device=str(device),
        attention_gates_enabled=True,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Predict tumor segmentation from MRI volume

    Args:
        file: NIfTI file (.nii or .nii.gz)

    Returns:
        Segmentation prediction as NIfTI file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = datetime.now()

    try:
        # Read uploaded file
        contents = await file.read()
        file_bytes = io.BytesIO(contents)

        # Load NIfTI
        nifti_img = nib.load(file_bytes)
        image_data = nifti_img.get_fdata()
        affine = nifti_img.affine

        logger.info(f"Loaded image with shape {image_data.shape}")

        # Preprocess
        preprocessed = preprocessor.preprocess(image_data)
        preprocessed = np.stack(
            [preprocessed] * 4, axis=0
        )  # Add channel dimension (4 modalities)
        preprocessed = torch.from_numpy(preprocessed).float().unsqueeze(0)  # Add batch
        preprocessed = preprocessed.to(device)

        # Inference
        with torch.no_grad():
            predictions = model(preprocessed)
            predictions = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()

        logger.info(f"Predictions shape: {predictions.shape}")

        # Save output
        output_filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nii.gz"
        output_path = OUTPUTS_DIR / output_filename

        nifti_pred = nib.Nifti1Image(predictions.astype(np.uint8), affine)
        nib.save(nifti_pred, output_path)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Prediction completed in {processing_time:.2f} seconds")

        return PredictionResponse(
            status="success",
            message="Segmentation completed successfully",
            processing_time=processing_time,
            output_filename=output_filename,
            predictions_url=f"/api/download/{output_filename}",
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/download/{filename}")
async def download_prediction(filename: str):
    """Download prediction result"""
    filepath = OUTPUTS_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(filepath, filename=filename, media_type="application/gzip")


@app.get("/api/classes")
async def get_classes():
    """Get available segmentation classes"""
    return {
        "0": "Background",
        "1": "Necrotic Core",
        "2": "Peritumoral Edema",
        "3": "Enhancing Tumor",
    }


@app.get("/docs")
async def swagger_ui():
    """Swagger UI documentation (auto-generated)"""
    pass


if __name__ == "__main__":
    # Create output directory
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run server
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    logger.info(f"Swagger UI: http://{API_HOST}:{API_PORT}/docs")

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info",
    )
