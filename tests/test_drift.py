"""Tests for drift detection module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.monitoring.drift import (
    compute_drift_metrics,
    load_embeddings,
    mmd_score,
    cosine_drift,
    ks_statistic,
    rbf_kernel,
)


def test_rbf_kernel_shape():
    x = np.random.randn(10, 128).astype(np.float32)
    y = np.random.randn(15, 128).astype(np.float32)
    k = rbf_kernel(x, y)
    assert k.shape == (10, 15)
    assert np.all(k >= 0)
    assert np.all(k <= 1)


def test_mmd_same_distribution():
    """MMD should be near zero for identical distributions."""
    rng = np.random.RandomState(42)
    x = rng.randn(100, 64).astype(np.float32)
    score = mmd_score(x, x)
    assert score < 0.01


def test_mmd_different_distribution():
    """MMD should be larger for shifted distributions."""
    rng = np.random.RandomState(42)
    x = rng.randn(100, 64).astype(np.float32)
    y = rng.randn(100, 64).astype(np.float32) + 5.0
    score_same = mmd_score(x, x)
    score_diff = mmd_score(x, y)
    assert score_diff > score_same * 5


def test_cosine_drift_same():
    rng = np.random.RandomState(42)
    x = rng.randn(100, 64).astype(np.float32)
    cd = cosine_drift(x, x)
    assert cd < 0.01


def test_cosine_drift_shifted():
    rng = np.random.RandomState(42)
    x = rng.randn(100, 64).astype(np.float32)
    y = rng.randn(100, 64).astype(np.float32) + 3.0
    cd = cosine_drift(x, y)
    assert cd > 0.0


def test_ks_statistic_range():
    rng = np.random.RandomState(42)
    x = rng.randn(100, 8).astype(np.float32)
    y = rng.randn(100, 8).astype(np.float32)
    ks = ks_statistic(x, y)
    assert 0.0 <= ks <= 1.0


def test_compute_drift_metrics_no_drift():
    rng = np.random.RandomState(42)
    x = rng.randn(500, 64).astype(np.float32)
    report = compute_drift_metrics(x, x + 1e-6, threshold=0.05)
    assert "mmd" in report
    assert "cosine_drift" in report
    assert "ks_statistic" in report
    assert not report["drift_detected"]


def test_compute_drift_metrics_drift_detected():
    rng = np.random.RandomState(42)
    x = rng.randn(500, 64).astype(np.float32)
    y = rng.randn(500, 64).astype(np.float32) + 2.0
    report = compute_drift_metrics(x, y, threshold=0.01)
    assert report["drift_detected"]


def test_compute_drift_metrics_dimension_mismatch():
    x = np.random.randn(10, 64).astype(np.float32)
    y = np.random.randn(10, 128).astype(np.float32)
    with pytest.raises(ValueError):
        compute_drift_metrics(x, y)


def test_load_embeddings():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        np.save(d / "img_001.npy", emb1)
        np.save(d / "img_002.npy", emb2)

        loaded = load_embeddings(str(d))
        assert loaded.shape == (2, 512)
        np.testing.assert_array_almost_equal(loaded[0], emb1)
        np.testing.assert_array_almost_equal(loaded[1], emb2)


def test_load_embeddings_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp)
