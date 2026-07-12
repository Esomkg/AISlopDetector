"""Drift detection for AISlopDetector.

Computes Maximum Mean Discrepancy (MMD) between reference and current
embedding distributions to detect data drift. Pure numpy — no heavy deps.

Usage:
    python -m src.monitoring.drift --reference data/embeddings/train --current data/embeddings/batch_20250701
"""

import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    """RBF kernel matrix between two sets of embedding vectors."""
    xx = np.sum(x * x, axis=1, keepdims=True)
    yy = np.sum(y * y, axis=1, keepdims=True)
    dists = xx + yy.T - 2 * np.dot(x, y.T)
    dists = np.maximum(dists, 0)

    if gamma is None:
        sigma = np.median(np.sqrt(dists[dists > 0])) if np.any(dists > 0) else 1.0
        gamma = 1.0 / (2.0 * sigma * sigma + 1e-8)

    return np.exp(-gamma * dists)


def mmd_score(reference: np.ndarray, current: np.ndarray) -> float:
    """Compute MMD² between reference and current embedding distributions."""
    m, n = reference.shape[0], current.shape[0]

    k_xx = rbf_kernel(reference, reference)
    k_yy = rbf_kernel(current, current)
    k_xy = rbf_kernel(reference, current)

    mmd2 = (
        k_xx.sum() / (m * m)
        + k_yy.sum() / (n * n)
        - 2 * k_xy.sum() / (m * n)
    )

    return float(np.sqrt(max(mmd2, 1e-12)))


def cosine_drift(reference: np.ndarray, current: np.ndarray) -> float:
    """Compute mean cosine distance between reference and current centroids."""
    ref_centroid = reference.mean(axis=0)
    cur_centroid = current.mean(axis=0)
    ref_norm = ref_centroid / (np.linalg.norm(ref_centroid) + 1e-8)
    cur_norm = cur_centroid / (np.linalg.norm(cur_centroid) + 1e-8)
    return float(1.0 - np.dot(ref_norm, cur_norm))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Mean KS statistic across all embedding dimensions."""
    ks_vals = []
    for dim in range(reference.shape[1]):
        ref_sorted = np.sort(reference[:, dim])
        cur_sorted = np.sort(current[:, dim])

        n1, n2 = len(ref_sorted), len(cur_sorted)
        cdf1 = np.searchsorted(ref_sorted, np.concatenate([ref_sorted, cur_sorted]), side="right") / n1
        cdf2 = np.searchsorted(cur_sorted, np.concatenate([ref_sorted, cur_sorted]), side="right") / n2
        ks_vals.append(np.max(np.abs(cdf1 - cdf2)))

    return float(np.mean(ks_vals))


def compute_drift_metrics(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """Compute full drift report between reference and current embeddings.

    Args:
        reference: NumPy array of shape (N, D) — reference embedding batch.
        current: NumPy array of shape (M, D) — current embedding batch.
        threshold: MMD threshold above which drift is considered significant.

    Returns:
        dict with mmd, cosine_drift, ks, threshold, and drift_detected (bool).
    """
    if reference.shape[1] != current.shape[1]:
        raise ValueError(
            f"Dimension mismatch: reference={reference.shape[1]}, "
            f"current={current.shape[1]}"
        )

    mmd = mmd_score(reference, current)
    cos = cosine_drift(reference, current)
    ks = ks_statistic(reference, current)

    return {
        "mmd": round(mmd, 6),
        "cosine_drift": round(cos, 6),
        "ks_statistic": round(ks, 6),
        "threshold": threshold,
        "drift_detected": mmd > threshold,
        "reference_size": reference.shape[0],
        "current_size": current.shape[0],
        "embedding_dim": reference.shape[1],
    }


def load_embeddings(directory: str) -> np.ndarray:
    """Load all .npy embedding files from a directory tree into a single array."""
    embeddings = []
    for npy_path in Path(directory).rglob("*.npy"):
        emb = np.load(npy_path)
        embeddings.append(emb.ravel())

    if not embeddings:
        raise FileNotFoundError(f"No .npy files found in {directory}")

    return np.stack(embeddings)


def generate_report(report: dict, output_path: Optional[str] = None) -> str:
    """Generate a human-readable drift report and return as string."""
    lines = [
        "=" * 50,
        "DRIFT DETECTION REPORT",
        "=" * 50,
        f"Reference samples: {report['reference_size']}",
        f"Current samples:   {report['current_size']}",
        f"Embedding dim:     {report['embedding_dim']}",
        "-" * 50,
        f"MMD:               {report['mmd']:.6f}",
        f"Cosine drift:      {report['cosine_drift']:.6f}",
        f"KS statistic:      {report['ks_statistic']:.6f}",
        f"Threshold:         {report['threshold']:.6f}",
        "-" * 50,
        f"Drift detected:    {report['drift_detected']}",
        "=" * 50,
    ]

    text = "\n".join(lines)
    print(text)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_path}")

    return text


def main():
    parser = argparse.ArgumentParser(description="AISlopDetector drift detection")
    parser.add_argument("--reference", required=True, help="Path to reference embeddings directory")
    parser.add_argument("--current", required=True, help="Path to current embeddings directory")
    parser.add_argument("--threshold", type=float, default=0.05, help="MMD drift threshold")
    parser.add_argument("--output", default=None, help="Path to save JSON report")
    args = parser.parse_args()

    reference = load_embeddings(args.reference)
    current = load_embeddings(args.current)

    print(f"Loaded reference: {reference.shape}")
    print(f"Loaded current:   {current.shape}")

    report = compute_drift_metrics(reference, current, args.threshold)
    generate_report(report, args.output)

    return report


if __name__ == "__main__":
    main()
