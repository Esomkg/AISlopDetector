"""Tests for champion/challenger evaluation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.evaluation.challenger import ChallengerEvaluator


def _make_dummy_checkpoint(path, backbone="efficientnet_b3", num_classes=2):
    """Create a dummy model checkpoint for testing."""
    from src.models.classifier import AISlopClassifier
    model = AISlopClassifier(num_classes=num_classes, backbone_name=backbone)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1, "val_accuracy": 0.85}, path)
    return path


def _make_dummy_cifake_dir(root, num_real=8, num_fake=8):
    """Create a dummy CIFAKE-style directory with images."""
    for split in ("train", "test"):
        for cls in ("REAL", "FAKE"):
            d = Path(root) / split / cls
            d.mkdir(parents=True)
            n = num_real if cls == "REAL" else num_fake
            for i in range(n):
                img = Image.new("RGB", (32, 32), color=(i * 30, 100, 150))
                img.save(d / f"img_{i}.jpg")
    return root


def _make_dummy_drift_dir(root, num_real=4, num_fake=4):
    """Create a dummy drift batch directory with CIFAKE-compatible structure."""
    for cls in ("REAL", "FAKE"):
        d = Path(root) / "test" / cls
        d.mkdir(parents=True)
        n = num_real if cls == "REAL" else num_fake
        for i in range(n):
            img = Image.new("RGB", (32, 32), color=(i * 40, 80, 200))
            img.save(d / f"img_{i}.jpg")
    return root


def test_challenger_loads_models():
    with tempfile.TemporaryDirectory() as tmp:
        champ_ckpt = Path(tmp) / "champ.pth"
        chall_ckpt = Path(tmp) / "chall.pth"
        _make_dummy_checkpoint(champ_ckpt)
        _make_dummy_checkpoint(chall_ckpt)

        evaluator = ChallengerEvaluator(
            str(champ_ckpt), str(chall_ckpt),
            backbone="efficientnet_b3", device="cpu",
        )
        assert evaluator.champion_model is not None
        assert evaluator.challenger_model is not None


def test_challenger_evaluates_model():
    from src.data.cifake_dataset import CIFAKE
    from src.data.transforms import get_val_transforms

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        holdout_dir = _make_dummy_cifake_dir(data_dir)
        transform = get_val_transforms(image_size=224)
        dataset = CIFAKE(data_root=str(holdout_dir), split="test", transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        champ_ckpt = Path(tmp) / "champ.pth"
        _make_dummy_checkpoint(champ_ckpt)

        evaluator = ChallengerEvaluator(
            str(champ_ckpt), str(champ_ckpt),
            backbone="efficientnet_b3", device="cpu",
        )

        metrics = evaluator.evaluate_model(evaluator.champion_model, loader)
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_challenger_run_both_splits():
    from src.data.cifake_dataset import CIFAKE
    from src.data.transforms import get_val_transforms

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        holdout_dir = str(_make_dummy_cifake_dir(base / "holdout"))
        drift_dir = str(_make_dummy_drift_dir(base / "drift"))

        transform = get_val_transforms(image_size=224)
        holdout_ds = CIFAKE(data_root=holdout_dir, split="test", transform=transform)
        drift_ds = CIFAKE(data_root=drift_dir, split="test", transform=transform)

        holdout_loader = torch.utils.data.DataLoader(holdout_ds, batch_size=4)
        drift_loader = torch.utils.data.DataLoader(drift_ds, batch_size=4)

        champ_ckpt = str(base / "champ.pth")
        chall_ckpt = str(base / "chall.pth")
        _make_dummy_checkpoint(champ_ckpt)
        _make_dummy_checkpoint(chall_ckpt)

        evaluator = ChallengerEvaluator(
            champ_ckpt, chall_ckpt,
            backbone="efficientnet_b3", device="cpu",
        )

        report = evaluator.run(holdout_loader, drift_loader)
        assert "champion" in report
        assert "challenger" in report
        assert "promoted" in report
        assert "decision" in report
        assert "holdout" in report["champion"]
        assert "drift" in report["champion"]


def test_challenger_compare_report_structure():
    evaluator = ChallengerEvaluator.__new__(ChallengerEvaluator)
    champ_metrics = {"accuracy": 0.80, "precision": 0.75, "recall": 0.77, "f1_score": 0.76, "roc_auc": 0.82}
    chall_metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.83, "f1_score": 0.82, "roc_auc": 0.88}
    result = evaluator._compare(champ_metrics, chall_metrics)
    assert "per_metric" in result
    assert "overall_winner" in result
    assert result["overall_winner"] == "challenger"
