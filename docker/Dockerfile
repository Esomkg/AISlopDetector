FROM apache/airflow:2.9.3-python3.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    timm \
    "mlflow" \
    pillow \
    pyyaml \
    numpy \
    scikit-learn \
    matplotlib \
    tqdm \
    python-dotenv \
    pandas \
    psycopg2-binary \
    open-clip-torch \
    kagglehub \
    fastapi \
    uvicorn \
    prometheus-client \
    azure-ai-ml \
    azure-identity
