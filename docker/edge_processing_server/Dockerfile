# Stage 1: Dependencies
FROM ubuntu:22.04 AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet
ENV DEBIAN_FRONTEND=noninteractive


# Install Python 3.11 and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        pkg-config \
        libhdf5-dev \
        build-essential \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set work directory for dependencies
WORKDIR /deps

# Copy only requirements file
COPY docker/edge_processing_server/requirements_edge_processing.txt .

# Create virtual environment and install dependencies
RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements_edge_processing.txt

# Stage 2: Final image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

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