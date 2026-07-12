"""Training script for the AISlop classifier."""

import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.cifake_dataset import CIFAKE
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.classifier import AISlopClassifier


def load_config(config_path="configs/training.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test with 64 images")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()

    cfg = load_config()
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    mlflow_cfg = cfg["mlflow"]
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    ckpt_cfg = cfg["checkpoint"]

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.smoke:
        train_cfg["num_workers"] = 0

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = get_train_transforms(
        image_size=train_cfg["image_size"],
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
    )
    val_transform = get_val_transforms(
        image_size=train_cfg["image_size"],
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
    )

    full_dataset = CIFAKE(data_root=data_cfg["raw_dir"], split="train", transform=train_transform)

    if args.smoke:
        full_dataset = torch.utils.data.Subset(full_dataset, range(min(64, len(full_dataset))))

    val_size = int(len(full_dataset) * data_cfg["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    model = AISlopClassifier(
        num_classes=train_cfg["num_classes"],
        backbone_name=train_cfg["backbone"],
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    Path(ckpt_cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    metrics_history = []

    with mlflow.start_run() as run:
        mlflow.log_params({
            "backbone": train_cfg["backbone"],
            "lr": train_cfg["learning_rate"],
            "batch_size": train_cfg["batch_size"],
            "epochs": train_cfg["epochs"],
            "image_size": train_cfg["image_size"],
        })

        for epoch in range(1, train_cfg["epochs"] + 1):
            print(f"\nEpoch {epoch}/{train_cfg['epochs']}")

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            metrics_history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            })

            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                ckpt_path = Path(ckpt_cfg["save_dir"]) / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(str(ckpt_path))
                print(f"  -> Saved best model (val_acc={val_acc:.4f})")

        with open("metrics/train_metrics.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
        mlflow.log_artifact("metrics/train_metrics.json")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
