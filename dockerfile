# -------------------------------
# 1. Base image with uv + Python
# -------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Ensure UTF-8 and non-interactive builds
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# -------------------------------
# 2. Copy project files
# .dockerignore will clean what we don't want
# -------------------------------
COPY . .

# -------------------------------
# 3. Install dependencies using uv
# This installs:
# - Your pyproject.toml dependencies
# - Editable package: rfdetr (from your [tool.uv.sources])
# -------------------------------
RUN uv sync --frozen

# -------------------------------
# 4. Expose Streamlit port
# -------------------------------
EXPOSE 8501

# -------------------------------
# 5. Production command
# -------------------------------
CMD ["uv", "run","-m", "streamlit", "run", "website_streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
