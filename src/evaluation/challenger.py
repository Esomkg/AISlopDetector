"""Champion/challenger evaluation harness for AISlopDetector.

Compares a newly trained model against the current production model
on a frozen holdout set and the latest drift-triggering batch.
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
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


class ChallengerEvaluator:
    def __init__(self, champion_checkpoint, challenger_checkpoint, backbone="efficientnet_b3", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss()

        self.champion_model = self.load_model(champion_checkpoint)
        self.challenger_model = self.load_model(challenger_checkpoint)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model = AISlopClassifier(num_classes=2, backbone_name=self.backbone, pretrained=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def evaluate_model(self, model, dataloader):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        running_loss = 0.0
        total = 0

        first_batch_size = None
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            if first_batch_size is None:
                first_batch_size = images.size(0)

            if images.size(0) != first_batch_size:
                continue

            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            loss = self.criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_probs[:, 1])),
            "loss": float(running_loss / total) if total > 0 else float("inf"),
        }

    def _compare(self, champ_metrics, chall_metrics):
        keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        better = {}
        champ_wins = 0
        chall_wins = 0

        for key in keys:
            if key not in champ_metrics or key not in chall_metrics:
                continue
            if key == "loss":
                if chall_metrics[key] < champ_metrics[key]:
                    better[key] = "challenger"
                    chall_wins += 1
                elif champ_metrics[key] < chall_metrics[key]:
                    better[key] = "champion"
                    champ_wins += 1
                else:
                    better[key] = "tie"
            else:
                if chall_metrics[key] > champ_metrics[key]:
                    better[key] = "challenger"
                    chall_wins += 1
                elif champ_metrics[key] > chall_metrics[key]:
                    better[key] = "champion"
                    champ_wins += 1
                else:
                    better[key] = "tie"

        if chall_wins > champ_wins:
            winner = "challenger"
        elif champ_wins > chall_wins:
            winner = "champion"
        else:
            winner = "tie"

        return {"per_metric": better, "overall_winner": winner, "score": f"{chall_wins}-{champ_wins}"}

    def run(self, holdout_loader, drift_loader):
        print("\nEvaluating CHAMPION on holdout...")
        champ_holdout = self.evaluate_model(self.champion_model, holdout_loader)
        print("Evaluating CHAMPION on drift batch...")
        champ_drift = self.evaluate_model(self.champion_model, drift_loader)

        print("Evaluating CHALLENGER on holdout...")
        chall_holdout = self.evaluate_model(self.challenger_model, holdout_loader)
        print("Evaluating CHALLENGER on drift batch...")
        chall_drift = self.evaluate_model(self.challenger_model, drift_loader)

        holdout_cmp = self._compare(champ_holdout, chall_holdout)
        drift_cmp = self._compare(champ_drift, chall_drift)

        holdout_winner = holdout_cmp["overall_winner"] == "challenger"
        drift_winner = drift_cmp["overall_winner"] == "challenger"
        promoted = holdout_winner and drift_winner

        if promoted:
            decision = "PROMOTED — Challenger beats champion on both holdout and drift splits."
        elif holdout_winner:
            decision = "REJECTED — Challenger wins on holdout but fails on drift. Possible overfitting."
        elif drift_winner:
            decision = "REJECTED — Challenger wins on drift but regresses on holdout. Unstable improvement."
        else:
            decision = "REJECTED — Challenger does not beat champion on either split."

        report = {
            "champion": {"holdout": champ_holdout, "drift": champ_drift},
            "challenger": {"holdout": chall_holdout, "drift": chall_drift},
            "holdout_comparison": holdout_cmp,
            "drift_comparison": drift_cmp,
            "promoted": promoted,
            "decision": decision,
        }
        return report


def print_report(report):
    print("\n" + "=" * 60)
    print("CHAMPION vs CHALLENGER EVALUATION REPORT")
    print("=" * 60)

    for split in ["holdout", "drift"]:
        print(f"\n--- {split.upper()} ---")
        print(f"{'Metric':<14} {'Champion':>12} {'Challenger':>12} {'Winner':>12}")
        print("-" * 52)
        keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "loss"]
        for key in keys:
            cmp_key = f"{split}_comparison"
            champ_val = report["champion"][split][key]
            chall_val = report["challenger"][split][key]
            winner = report[cmp_key]["per_metric"].get(key, "tie")

            if key == "loss":
                fmt_champ = f"{champ_val:>12.4f}"
                fmt_chall = f"{chall_val:>12.4f}"
            else:
                fmt_champ = f"{champ_val:>12.4f}"
                fmt_chall = f"{chall_val:>12.4f}"
            print(f"{key:<14} {fmt_champ} {fmt_chall} {winner:>12}")

    print("\n" + "-" * 60)
    print(f"Holdout overall winner:  {report['holdout_comparison']['overall_winner']}")
    print(f"Drift overall winner:    {report['drift_comparison']['overall_winner']}")
    print(f"Promoted:                {report['promoted']}")
    print(f"\nDECISION: {report['decision']}")
    print("=" * 60 + "\n")


def run_challenger_evaluation(
    champion_ckpt,
    challenger_ckpt,
    holdout_dir,
    drift_dir,
    config_path="configs/training.yaml",
):
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    mlflow_cfg = cfg["mlflow"]
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])

    backbone = train_cfg["backbone"]

    val_transform = get_val_transforms(
        image_size=train_cfg["image_size"],
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
    )

    holdout_dataset = CIFAKE(data_root=holdout_dir, split="test", transform=val_transform)
    drift_dataset = CIFAKE(data_root=drift_dir, split="test", transform=val_transform)

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    drift_loader = DataLoader(
        drift_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    evaluator = ChallengerEvaluator(
        champion_checkpoint=champion_ckpt,
        challenger_checkpoint=challenger_ckpt,
        backbone=backbone,
    )

    report = evaluator.run(holdout_loader, drift_loader)

    print_report(report)

    Path("metrics").mkdir(parents=True, exist_ok=True)
    report_path = "metrics/challenger_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    with mlflow.start_run(run_name="challenger_evaluation"):
        mlflow.log_param("champion_checkpoint", champion_ckpt)
        mlflow.log_param("challenger_checkpoint", challenger_ckpt)
        mlflow.log_param("holdout_dir", holdout_dir)
        mlflow.log_param("drift_dir", drift_dir)

        for split in ["holdout", "drift"]:
            champ_metrics = report["champion"][split]
            chall_metrics = report["challenger"][split]
            for key, val in champ_metrics.items():
                mlflow.log_metric(f"champion_{split}_{key}", val)
            for key, val in chall_metrics.items():
                mlflow.log_metric(f"challenger_{split}_{key}", val)

        mlflow.log_metric("promoted", int(report["promoted"]))
        mlflow.log_artifact(report_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AISlopDetector champion/challenger evaluation")
    parser.add_argument("--champion", required=True, help="Path to champion checkpoint")
    parser.add_argument("--challenger", required=True, help="Path to challenger checkpoint")
    parser.add_argument("--holdout-dir", required=True, help="Path to frozen holdout dataset")
    parser.add_argument("--drift-dir", required=True, help="Path to drift-triggering batch dataset")
    parser.add_argument("--config", default="configs/training.yaml", help="Path to config YAML")
    args = parser.parse_args()

    run_challenger_evaluation(
        champion_ckpt=args.champion,
        challenger_ckpt=args.challenger,
        holdout_dir=args.holdout_dir,
        drift_dir=args.drift_dir,
        config_path=args.config,
    )
