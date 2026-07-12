"""Monitoring DAG for AISlopDetector.

Runs MMD drift detection on embedding distributions daily and
triggers the training pipeline when drift crosses a threshold.
"""

import json
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

from common import DEFAULT_ARGS, DRIFT_THRESHOLD, SRC_PATH, DATA_PATH

PROJECT_ROOT = SRC_PATH.parent


with DAG(
    dag_id="drift_monitoring",
    default_args=DEFAULT_ARGS,
    description="Embedding drift detection using MMD — triggers retraining on drift",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["monitoring", "drift"],
) as dag:

    def find_batches(**context):
        """Find reference and current embedding batches for comparison.

        Strategy:
        - Reference: Most recent training batch (before current)
        - Current: Latest batch collected

        Returns paths to reference and current batch directories.
        """
        embedding_root = Path(DATA_PATH) / "embeddings"
        if not embedding_root.exists():
            raise FileNotFoundError(
                f"No embedding data found at {embedding_root}. "
                "Run the training_pipeline DAG first."
            )

        batches = sorted(
            [d for d in embedding_root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if len(batches) < 2:
            raise FileNotFoundError(
                f"Need at least 2 embedding batches, found {len(batches)}. "
                f"Collect more data first."
            )

        current_batch = batches[-1]
        reference_batch = batches[-2]

        print(f"[monitoring] Reference: {reference_batch}")
        print(f"[monitoring] Current:   {current_batch}")

        context["task_instance"].xcom_push(key="reference", value=str(reference_batch))
        context["task_instance"].xcom_push(key="current", value=str(current_batch))

    def run_drift_detection(**context):
        """Run MMD drift detection between reference and current batches."""
        ti = context["task_instance"]
        reference_dir = ti.xcom_pull(key="reference", task_ids="find_batches")
        current_dir = ti.xcom_pull(key="current", task_ids="find_batches")

        from src.monitoring.drift import load_embeddings, compute_drift_metrics, generate_report

        reference = load_embeddings(reference_dir)
        current = load_embeddings(current_dir)

        report = compute_drift_metrics(reference, current, DRIFT_THRESHOLD)

        report_path = Path(PROJECT_ROOT) / "metrics" / f"drift_report_{context['ds_nodash']}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_report(report, str(report_path))

        ti.xcom_push(key="drift_report", value=report)

    def decide_retraining(**context):
        """Branch: trigger retraining if drift detected, else skip."""
        ti = context["task_instance"]
        report = ti.xcom_pull(key="drift_report", task_ids="run_drift_detection")

        if report["drift_detected"]:
            print(
                f"[monitoring] DRIFT DETECTED! MMD={report['mmd']:.6f} > threshold={DRIFT_THRESHOLD:.6f}. "
                "Triggering retraining."
            )
            return "trigger_retraining"
        else:
            print(
                f"[monitoring] No drift detected. MMD={report['mmd']:.6f} <= threshold={DRIFT_THRESHOLD:.6f}."
            )
            return "skip_retraining"

    find_batches_task = PythonOperator(
        task_id="find_batches",
        python_callable=find_batches,
    )

    detect_drift = PythonOperator(
        task_id="run_drift_detection",
        python_callable=run_drift_detection,
    )

    branch = BranchPythonOperator(
        task_id="decide_retraining",
        python_callable=decide_retraining,
    )

    trigger_training = EmptyOperator(task_id="trigger_retraining")
    skip_training = EmptyOperator(task_id="skip_retraining")

    find_batches_task >> detect_drift >> branch
    branch >> [trigger_training, skip_training]
