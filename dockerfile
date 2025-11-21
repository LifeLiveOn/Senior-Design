# Use slim Python base (no multi-stage to avoid layer duplication)
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system deps + uv in one layer, clean up immediately
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY RF-DETR-model_modified/rf-detr-modifications ./RF-DETR-model_modified/rf-detr-modifications

# Install Python packages and clean up in same layer
RUN uv sync --frozen --no-dev && \
    find /app/.venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name '*.pyc' -delete && \
    find /app/.venv -type f -name '*.pyo' -delete

# Copy application code
COPY backend ./backend
COPY cloud ./cloud
COPY model_utils.py .

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
