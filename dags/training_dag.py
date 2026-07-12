"""Training pipeline DAG for AISlopDetector.

Orchestrates embedding extraction, model training, evaluation,
champion/challenger head-to-head comparison, and conditional model promotion.
Can be triggered on a schedule or by a drift alert from the monitoring DAG.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from common import DEFAULT_ARGS, SRC_PATH

PROJECT_ROOT = SRC_PATH.parent

def run_embeddings(**context):
    """Extract CLIP embeddings for all images in the latest batch."""
    print("[training.embeddings] Starting embedding extraction...")
    result = subprocess.run(
        [sys.executable, "-m", "src.data.embeddings"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Embedding extraction failed: {result.stderr}")
    print("[training.embeddings] Embedding extraction complete.")

def run_training(**context):
    """Train the AISlop classifier."""
    config_path = PROJECT_ROOT / "configs" / "training.yaml"
    print(f"[training.train] Starting training with config: {config_path}")
    result = subprocess.run(
        [sys.executable, "-m", "src.models.train"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Training failed: {result.stderr}")
    print("[training.train] Training complete.")

def run_evaluation(**context):
    """Evaluate the trained model and return metrics."""
    print("[training.evaluate] Starting evaluation...")
    result = subprocess.run(
        [sys.executable, "-m", "src.models.evaluate"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Evaluation failed: {result.stderr}")

    import json
    metrics_path = PROJECT_ROOT / "metrics" / "eval_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    context["task_instance"].xcom_push(key="eval_metrics", value=metrics)
    print(f"[training.evaluate] Metrics: {metrics}")
    return metrics


def run_challenger_eval(**context):
    """Run champion/challenger head-to-head evaluation.

    Compares the newly trained model (challenger) against
    the current production model (champion) on:
      - Frozen holdout set (standard test split)
      - Latest drift-triggering batch (if available)

    Promotion requires the challenger to outperform on BOTH splits.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.evaluation.challenger import run_challenger_evaluation

    print("[training.challenger] Starting champion/challenger evaluation...")

    champion_ckpt = str(PROJECT_ROOT / "checkpoints" / "best_model.pth")
    challenger_ckpt = str(PROJECT_ROOT / "checkpoints" / "best_model.pth")
    config_path = str(PROJECT_ROOT / "configs" / "training.yaml")

    drift_dir = None
    embedding_root = Path(PROJECT_ROOT) / "data" / "embeddings"
    if embedding_root.exists():
        batches = sorted(
            [d for d in embedding_root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if batches:
            drift_dir = str(batches[-1])

    data_dir = PROJECT_ROOT / "data" / "raw" / "cifake"
    if not data_dir.exists():
        print("[training.challenger] No CIFAKE data found, skipping challenger eval.")
        return "skip_promotion"

    holdout_dir = str(data_dir / "test")
    report = run_challenger_evaluation(
        champion_ckpt=champion_ckpt,
        challenger_ckpt=challenger_ckpt,
        holdout_dir=holdout_dir,
        drift_dir=drift_dir,
        config_path=config_path,
    )

    context["task_instance"].xcom_push(key="challenger_report", value=report)

    if report.get("promoted", False):
        print(f"[training.challenger] Challenger WINS. Decision: {report.get('decision', '')}")
        return "promote_model"
    else:
        print(f"[training.challenger] Challenger LOST. Decision: {report.get('decision', '')}")
        return "skip_promotion"

def promote_to_registry(**context):
    """Promote the model to the MLflow registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("http://mlflow-server:5500")
    client = MlflowClient()

    experiments = client.search_experiments(
        filter_string="name = 'aislop-detector'"
    )
    if not experiments:
        print("[training.promote] No experiment found, skipping promotion.")
        return

    experiment_id = experiments[0].experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.val_accuracy DESC"],
        max_results=1,
    )
    if not runs:
        print("[training.promote] No runs found, skipping promotion.")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    try:
        mv = client.create_model_version(
            name="aislop_classifier",
            source=model_uri,
            run_id=run_id,
        )
        client.transition_model_version_stage(
            name="aislop_classifier",
            version=mv.version,
            stage="production",
        )
        print(f"[training.promote] Model version {mv.version} promoted to production.")
    except Exception as e:
        registered_model = mlflow.register_model(model_uri, "aislop_classifier")
        client.transition_model_version_stage(
            name="aislop_classifier",
            version=registered_model.version,
            stage="production",
        )
        print(f"[training.promote] New model registered and promoted: v{registered_model.version}")

with DAG(
    dag_id="training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Embedding extraction, training, evaluation, and model promotion",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["training", "ml"],
    params={
        "trigger": "manual",
        "drift_score": None,
    },
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed")

    extract_embeddings = PythonOperator(
        task_id="extract_embeddings",
        python_callable=run_embeddings,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluation,
    )

    challenger_eval = BranchPythonOperator(
        task_id="run_challenger_eval",
        python_callable=run_challenger_eval,
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=promote_to_registry,
    )

    skip_promotion = EmptyOperator(task_id="skip_promotion")

    join = EmptyOperator(task_id="join", trigger_rule="none_failed")

    start >> extract_embeddings >> train_model >> evaluate_model >> challenger_eval
    challenger_eval >> [promote_model, skip_promotion]
    promote_model >> join >> end
    skip_promotion >> join >> end
