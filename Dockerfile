# Multi-stage Dockerfile for Brain Tumour Segmentation Project

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim-bullseye as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as builder

# Copy requirements
COPY requirements.txt .

# Create wheels for faster installation
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir -r requirements.txt

# Stage 3: API Server
FROM base as api

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set Python path
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY . /app

# Install package in development mode
RUN pip install --user --no-cache-dir -e .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Jupyter Lab for research/development
FROM base as jupyter

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set Python path
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY . /app

# Install package
RUN pip install --user --no-cache-dir -e .

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

# Stage 5: Training container
FROM base as training

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set Python path
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY . /app

# Install package
RUN pip install --user --no-cache-dir -e .

# Set entry point for training
ENTRYPOINT ["python", "-m", "training.train"]
