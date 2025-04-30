# Stage 1: Dependencies
FROM python:3.8-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet

# Set work directory for dependencies
WORKDIR /deps

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libhdf5-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements file
COPY docker/edge_processing_server/requirements_edge_processing.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements_edge_processing.txt

# Stage 2: Final image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY edge_processing/ ./edge_processing/
COPY dataset/ ./dataset/
COPY utils/ ./utils/
COPY opa/ ./opa/
COPY kaggle_credentials/ ./kaggle_credentials/

# Create directories for model storage
# RUN mkdir -p /app/models/mobilenet /app/models/tinybert /app/models/t5

# Expose necessary ports
EXPOSE 8002

# Change the CMD to ENTRYPOINT and CMD
ENTRYPOINT ["python3", "edge_processing/aggregator.py"]