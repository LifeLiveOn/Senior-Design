FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---- Minimal system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH"

# ---- Copy RF-DETR FIRST (required for editable install) ----
COPY RF-DETR-model_modified/rf-detr-modifications \
    /app/RF-DETR-model_modified/rf-detr-modifications

# ---- Copy dependency metadata ----
COPY pyproject.toml uv.lock ./

# ---- Install Python deps ----
RUN uv sync --no-dev

# ---- App code ----
COPY backend /app/backend
WORKDIR /app/backend

EXPOSE 8000

CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120"]
