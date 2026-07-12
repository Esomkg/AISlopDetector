#!/bin/bash
set -e

echo "=== AISlopDetector Airflow Init ==="

echo "Creating mlflow database if not exists..."
PGPASSWORD=airflow psql -h postgres -U airflow -d airflow -tc \
    "SELECT 1 FROM pg_database WHERE datname = 'mlflow'" | grep -q 1 || \
    PGPASSWORD=airflow psql -h postgres -U airflow -d airflow -c "CREATE DATABASE mlflow"

echo "Running Airflow DB migrations..."
airflow db migrate

echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin \
    || echo "Admin user already exists, skipping"

echo "Airflow init complete."
