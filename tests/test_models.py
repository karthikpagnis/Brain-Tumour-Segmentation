"""
Unit tests for Brain Tumor Segmentation Project
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from models.unet_attention import AttentionUNet3D
from models.loss_functions import DiceBCELoss, DiceLoss
from training.metrics import SegmentationMetrics, MetricAggregator
from config import NUM_INPUT_CHANNELS, NUM_CLASSES


class TestModel:
    """Test model architecture and functionality"""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        return AttentionUNet3D().to(device)

    def test_model_initialization(self, model):
        """Test model can be instantiated"""
        assert model is not None
        assert isinstance(model, AttentionUNet3D)

    def test_forward_pass(self, model, device):
        """Test forward pass with correct input/output shapes"""
        batch_size = 2
        x = torch.randn(batch_size, NUM_INPUT_CHANNELS, 32, 32, 32).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, NUM_CLASSES, 32, 32, 32)

    def test_model_parameters(self, model):
        """Test model has trainable parameters"""
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params > 1_000_000  # Should have several million parameters

    def test_gradient_flow(self, model, device):
        """Test gradients can flow through model"""
        x = torch.randn(1, NUM_INPUT_CHANNELS, 32, 32, 32).to(device)
        targets = torch.randint(0, NUM_CLASSES, (1, 32, 32, 32)).to(device)

        loss_fn = DiceBCELoss().to(device)

        output = model(x)
        loss, _ = loss_fn(output, targets)

        loss.backward()

        # Check gradients exist
        has_grads = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads, "No gradients computed"


class TestLossFunctions:
    """Test loss function computation"""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dice_loss(self, device):
        """Test Dice loss computation"""
        dice = DiceLoss().to(device)

        predictions = torch.randn(2, 4, 32, 32, 32).to(device)
        targets = torch.randint(0, 4, (2, 32, 32, 32)).to(device)

        loss = dice(predictions, targets)

        assert loss.item() >= 0
        assert loss.item() <= 1.0

    def test_composite_loss(self, device):
        """Test composite Dice-BCE loss"""
        loss_fn = DiceBCELoss().to(device)

        predictions = torch.randn(2, 4, 32, 32, 32).to(device)
        targets = torch.randint(0, 4, (2, 32, 32, 32)).to(device)

        loss, components = loss_fn(predictions, targets)

        assert loss.item() > 0
        assert "dice" in components
        assert "bce" in components


class TestMetrics:
    """Test evaluation metrics"""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dice_score(self, device):
        """Test Dice score computation"""
        # Perfect prediction
        pred = torch.ones(2, 32, 32, 32, dtype=torch.int64).to(device)
        target = torch.ones(2, 32, 32, 32, dtype=torch.int64).to(device)

        dice = SegmentationMetrics.dice_score(pred, target)
        assert dice == 1.0

        # No overlap
        pred = torch.zeros(2, 32, 32, 32, dtype=torch.int64).to(device)
        target = torch.ones(2, 32, 32, 32, dtype=torch.int64).to(device)

        dice = SegmentationMetrics.dice_score(pred, target)
        assert dice < 0.01

    def test_iou_score(self, device):
        """Test IoU score computation"""
        pred = torch.ones(2, 32, 32, 32, dtype=torch.int64).to(device)
        target = torch.ones(2, 32, 32, 32, dtype=torch.int64).to(device)

        iou = SegmentationMetrics.iou_score(pred, target)
        assert iou == 1.0

    def test_metric_aggregator(self, device):
        """Test metric aggregation across batches"""
        aggregator = MetricAggregator(NUM_CLASSES)

        for _ in range(3):
            pred = torch.randint(0, NUM_CLASSES, (2, 32, 32, 32)).to(device)
            target = torch.randint(0, NUM_CLASSES, (2, 32, 32, 32)).to(device)
            aggregator.update(pred, target)

        scores = aggregator.get_scores()

        assert "dice_mean" in scores
        assert "iou_mean" in scores
        assert "f1_mean" in scores
        assert aggregator.batch_count == 3


class TestDataPipeline:
    """Test data loading and preprocessing"""

    def test_mock_dataset_creation(self):
        """Test mock dataset can be created"""
        try:
            from scripts.download_data import create_mock_data

            with tempfile.TemporaryDirectory() as tmpdir:
                create_mock_data(num_cases=2, output_dir=tmpdir)

                # Verify files created
                import os

                files = []
                for root, dirs, filenames in os.walk(tmpdir):
                    files.extend(filenames)

                assert len(files) > 0
                assert any(".nii.gz" in f for f in files)

        except ImportError:
            pytest.skip("Dataset module not available")


class TestIntegration:
    """Integration tests combining multiple components"""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step(self, device):
        """Test complete training step"""
        # Setup
        model = AttentionUNet3D().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = DiceBCELoss().to(device)

        # Create batch
        batch_size = 1
        images = torch.randn(batch_size, NUM_INPUT_CHANNELS, 32, 32, 32).to(device)
        targets = torch.randint(0, NUM_CLASSES, (batch_size, 32, 32, 32)).to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss, components = loss_fn(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Verify
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_model_prediction(self, device):
        """Test end-to-end prediction"""
        model = AttentionUNet3D().to(device)
        model.eval()

        # Create input
        x = torch.randn(1, NUM_INPUT_CHANNELS, 64, 64, 64).to(device)

        # Predict
        with torch.no_grad():
            output = model(x)
            predictions = torch.argmax(output, dim=1)

        # Verify
        assert predictions.shape == (1, 64, 64, 64)
        assert predictions.min() >= 0
        assert predictions.max() < NUM_CLASSES


def test_config_import():
    """Test configuration can be imported"""
    from config import (
        NUM_CLASSES,
        NUM_INPUT_CHANNELS,
        BATCH_SIZE,
        LEARNING_RATE,
    )

    assert NUM_CLASSES == 4
    assert NUM_INPUT_CHANNELS == 4
    assert BATCH_SIZE > 0
    assert LEARNING_RATE > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
