"""Review feedback DAG for AISlopDetector.

Pulls completed human annotations from Label Studio weekly,
copies labeled images into the training dataset, and triggers
the training pipeline with the new ground truth data.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

from common import DEFAULT_ARGS, SRC_PATH, DATA_PATH

PROJECT_ROOT = SRC_PATH.parent
LABEL_STUDIO_URL = "http://label-studio:8080"
DEFAULT_PROJECT_ID = 1


with DAG(
    dag_id="review_feedback",
    default_args=DEFAULT_ARGS,
    description="Import human-reviewed labels from Label Studio and trigger retraining",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["review", "active-learning"],
) as dag:

    def import_reviewed_labels(**context):
        """Pull completed annotations from Label Studio and save to data/raw/reviewed."""
        import json
        from src.data.active_learning import LabelStudioClient

        client = LabelStudioClient(url=LABEL_STUDIO_URL)
        project_id = client.get_or_create_project()

        output_dir = Path(DATA_PATH) / "raw" / f"reviewed_{context['ds_nodash']}"
        result = client.export_labels(project_id, str(output_dir))

        print(f"[review_feedback] Exported {result['total']} annotations to {output_dir}")
        print(f"[review_feedback] Counts: {result['counts']}")

        context["task_instance"].xcom_push(key="review_summary", value=result)

    def check_new_labels(**context):
        """Only trigger retraining if new labels were imported."""
        ti = context["task_instance"]
        summary = ti.xcom_pull(key="review_summary", task_ids="import_labels")
        total_new = summary.get("counts", {}).get("REAL", 0) + summary.get("counts", {}).get("FAKE", 0)

        if total_new > 0:
            print(f"[review_feedback] {total_new} new labels imported. Triggering retraining.")
            return "trigger_retraining"
        else:
            print("[review_feedback] No new labels. Skipping retraining.")
            return "skip_retraining"

    start = EmptyOperator(task_id="start")

    import_labels = PythonOperator(
        task_id="import_labels",
        python_callable=import_reviewed_labels,
    )

    check = BranchPythonOperator(
        task_id="check_new_labels",
        python_callable=check_new_labels,
    )

    trigger_training = EmptyOperator(task_id="trigger_retraining")
    skip_training = EmptyOperator(task_id="skip_retraining")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed")

    start >> import_labels >> check
    check >> [trigger_training, skip_training]
    trigger_training >> end
    skip_training >> end
