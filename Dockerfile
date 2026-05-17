# ── Stage 1: Base Python image ──────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_URL=http://localhost:8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose ports: FastAPI=8000, Streamlit=8501
EXPOSE 8000 8501

# Default: Run FastAPI (override with CMD for Streamlit)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
