"""Evaluation script for the AISlop classifier."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cifake_dataset import CIFAKE
from src.data.transforms import get_val_transforms
from src.models.classifier import AISlopClassifier


def load_config(config_path="configs/training.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    cfg = load_config()
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    mlflow_cfg = cfg["mlflow"]
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AISlopClassifier(
        num_classes=train_cfg["num_classes"],
        backbone_name=train_cfg["backbone"],
        pretrained=False,
    ).to(device)

    ckpt_path = Path("checkpoints") / "best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_transform = get_val_transforms(
        image_size=train_cfg["image_size"],
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
    )
    test_dataset = CIFAKE(data_root=data_cfg["raw_dir"], split="test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_probs[:, 1])

    print("\n" + "=" * 45)
    print(f"{'Metric':<15} {'Value':>10}")
    print("=" * 45)
    print(f"{'Accuracy':<15} {accuracy:>10.4f}")
    print(f"{'Precision':<15} {precision:>10.4f}")
    print(f"{'Recall':<15} {recall:>10.4f}")
    print(f"{'F1 Score':<15} {f1:>10.4f}")
    print(f"{'ROC-AUC':<15} {roc_auc:>10.4f}")
    print("=" * 45 + "\n")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }

    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = "metrics/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params({
            "backbone": train_cfg["backbone"],
            "checkpoint": str(ckpt_path),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("metrics/eval_metrics.json")
        mlflow.log_artifact(cm_path)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
