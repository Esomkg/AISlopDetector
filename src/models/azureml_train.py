"""Azure ML training entry point for AISlopDetector.

This script is executed inside an Azure ML Compute job container.
It automates downloading CIFAKE data, training the classifier,
and registering the model in the MLflow-backed Azure ML registry.

Run as:
  python azureml_train.py --data-dir /mnt/data --epochs 10

Environment variables (set by Azure ML):
  AZUREML_OUTPUT_DIR  - where to save checkpoints and metrics
  MLFLOW_TRACKING_URI - MLflow tracking server URL
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download_data_on_azure(output_dir: str):
    """Download CIFAKE dataset at runtime (Azure ML compute has internet)."""
    output_dir = Path(output_dir)
    required = [
        output_dir / "train" / "REAL",
        output_dir / "train" / "FAKE",
        output_dir / "test" / "REAL",
        output_dir / "test" / "FAKE",
    ]

    if all(d.is_dir() and any(d.glob("*.jpg")) for d in required):
        print(f"Data already exists at {output_dir}, skipping download.")
        return str(output_dir)

    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub not installed. Install with: pip install kagglehub"
        )

    print("Downloading CIFAKE dataset on Azure ML compute...")
    cache_path = Path(
        kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    )

    import shutil
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(cache_path, output_dir)

    print(f"Dataset ready at {output_dir}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="AISlopDetector Azure ML training")
    parser.add_argument("--data-dir", default="data/raw/cifake", help="Path to CIFAKE dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", default="efficientnet_b3", help="Model backbone")
    parser.add_argument("--config", default=None, help="Override config file path")
    parser.add_argument("--download", action="store_true", help="Download CIFAKE before training")
    args = parser.parse_args()

    output_dir = os.environ.get("AZUREML_OUTPUT_DIR", str(PROJECT_ROOT))

    if args.download:
        data_dir = download_data_on_azure(args.data_dir)
    else:
        data_dir = args.data_dir

    from src.data.cifake_dataset import CIFAKE
    from src.data.transforms import get_train_transforms, get_val_transforms
    from src.models.classifier import AISlopClassifier

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = get_train_transforms(image_size=224)
    val_transform = get_val_transforms(image_size=224)

    full_dataset = CIFAKE(data_root=data_dir, split="train", transform=train_transform)

    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {train_size}, Val: {val_size}")

    model = AISlopClassifier(num_classes=2, backbone_name=args.backbone, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5500"))
    mlflow.set_experiment("aislop-detector")

    best_val_acc = 0.0
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(output_dir) / "best_model.pth"

    with mlflow.start_run() as run:
        mlflow.log_params({
            "backbone": args.backbone,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "compute": "azure-ml",
        })

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = train_correct = train_total = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            scheduler.step()

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            train_l = train_loss / train_total
            val_l = val_loss / val_total

            mlflow.log_metrics({
                "train_loss": train_l, "train_accuracy": train_acc,
                "val_loss": val_l, "val_accuracy": val_acc,
            }, step=epoch)

            print(f"Epoch {epoch}: train_loss={train_l:.4f} train_acc={train_acc:.4f}  val_loss={val_l:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_accuracy": val_acc}, ckpt_path)
                mlflow.log_artifact(str(ckpt_path))
                print(f"  -> Saved best model (acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
