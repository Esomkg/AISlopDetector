"""Tests for inference serving."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.serving.model_server import ModelService, CLASS_LABELS


def _make_dummy_checkpoint(path, backbone="efficientnet_b3"):
    """Create a dummy model checkpoint for testing."""
    from src.models.classifier import AISlopClassifier
    model = AISlopClassifier(num_classes=2, backbone_name=backbone)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1, "val_accuracy": 0.90}, path)
    return path


def _dummy_image():
    """Return a small dummy PIL image."""
    return Image.new("RGB", (224, 224), color=(100, 150, 200))


def test_model_service_initializes():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "model.pth"
        _make_dummy_checkpoint(ckpt)
        service = ModelService(checkpoint_path=str(ckpt), device="cpu")
        assert service.model is not None
        assert service.device.type == "cpu"


def test_model_service_predict_structure():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "model.pth"
        _make_dummy_checkpoint(ckpt)
        service = ModelService(checkpoint_path=str(ckpt), device="cpu")

        img = _dummy_image()
        result = service.predict(img)

        assert "predicted_class" in result
        assert result["predicted_class"] in CLASS_LABELS.values()
        assert "confidence" in result
        assert "probabilities" in result
        assert "REAL" in result["probabilities"]
        assert "FAKE" in result["probabilities"]
        assert 0.0 <= result["confidence"] <= 1.0
        probs = result["probabilities"]
        assert abs(probs["REAL"] + probs["FAKE"] - 1.0) < 0.01


def test_model_service_predict_batch():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "model.pth"
        _make_dummy_checkpoint(ckpt)
        service = ModelService(checkpoint_path=str(ckpt), device="cpu")

        images = [_dummy_image() for _ in range(4)]
        results = service.predict_batch(images)

        assert len(results) == 4
        for result in results:
            assert "predicted_class" in result
            assert "confidence" in result
