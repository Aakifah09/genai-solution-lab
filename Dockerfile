# GenAI Solution Lab: Streamlit dashboard + CLI in one CPU image.
#   docker compose up --build            -> http://localhost:8501
#   docker compose run --rm lab python cli.py train --model vae --epochs 2
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GENAI_LAB_DATA_DIR=/app/data \
    GENAI_LAB_OUTPUT_DIR=/app/outputs \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

# CPU-only PyTorch first (the default PyPI wheel bundles ~2 GB of CUDA libraries),
# then the rest; pip sees torch/torchvision already satisfied.
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

RUN useradd --create-home --uid 1000 lab \
 && mkdir -p /app/data /app/outputs /app/.cache/torch \
 && chown -R lab:lab /app
COPY --chown=lab:lab . .
USER lab

VOLUME ["/app/data", "/app/outputs", "/app/.cache/torch"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
