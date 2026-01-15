# =====================================================
# Production Django Backend – GPU (ONNX Runtime)
# =====================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# -----------------------------------------------------
# System dependencies (Python 3.10 from Ubuntu)
# -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libpq5 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*



# -----------------------------------------------------
# Install uv
# -----------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# -----------------------------------------------------
# Copy dependency metadata
# -----------------------------------------------------
COPY pyproject.toml uv.lock ./
COPY RF-DETR-model_modified/rf-detr-modifications \
    ./RF-DETR-model_modified/rf-detr-modifications

# -----------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# -----------------------------------------------------
# Copy application code
# -----------------------------------------------------
COPY backend ./backend
WORKDIR /app/backend

EXPOSE 8000

CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
