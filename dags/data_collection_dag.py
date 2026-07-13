"""Data collection DAG for AISlopDetector.

Runs weekly to collect new AI-generated and real images from various sources,
tag them with generator metadata, and store them for downstream processing.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from common import DEFAULT_ARGS, DATA_PATH

with DAG(
    dag_id="data_collection",
    default_args=DEFAULT_ARGS,
    description="Weekly data collection from generator sources",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["data", "collection"],
) as dag:

    def collect_images(**context):
        """Collect new images from configured sources (stub for local dev)."""
        batch_date = context["ds_nodash"]
        batch_dir = Path(DATA_PATH) / "raw" / f"batch_{batch_date}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        generators = ["midjourney_v6", "dalle3", "sdxl", "flux_pro"]

        for gen in generators:
            gen_dir = batch_dir / gen
            gen_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "generator_name": gen,
                "collection_date": context["ds"],
                "source": f"stub_{gen}_api",
                "image_count": 0,
                "note": "Stub — replace with actual scraper in production",
            }

            import json
            with open(gen_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"[data_collection] Batch {batch_date} created at {batch_dir}")
        print(f"[data_collection] Generators: {generators}")
        print(f"[data_collection] NOTE: This is a stub. Replace with real scrapers (Phase 5+).")

    def tag_real_images(**context):
        """Collect real images from Unsplash API."""
        batch_date = context["ds_nodash"]
        real_dir = Path(DATA_PATH) / "raw" / f"batch_{batch_date}" / "REAL"

        try:
            from src.data.scrapers.unsplash_scraper import UnsplashScraper

            scraper = UnsplashScraper(output_dir=real_dir)
            count = scraper.scrape(max_images=50)
            print(f"[data_collection] Downloaded {count} real images from Unsplash to {real_dir}")
        except ImportError:
            print("[data_collection] UnsplashScraper not available — falling back to stub")
            real_dir.mkdir(parents=True, exist_ok=True)
            import json
            with open(real_dir / "metadata.json", "w") as f:
                json.dump({"source": "unsplash_api", "count": 0, "note": "Stub — set UNSPLASH_ACCESS_KEY"}, f, indent=2)
        except Exception as e:
            print(f"[data_collection] Unsplash error: {e}")

    collect_generated = PythonOperator(
        task_id="collect_generated_images",
        python_callable=collect_images,
    )

    collect_real = PythonOperator(
        task_id="collect_real_images",
        python_callable=tag_real_images,
    )

    collect_generated >> collect_real
