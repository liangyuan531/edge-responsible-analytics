#!/bin/bash

# Script to download, setup and deploy the edge-responsible-analytics project

# --- Configuration ---
REPO_URL="https://github.com/pipeedge/edge-responsible-analytics/archive/refs/tags/release.zip"
REPO_NAME="edge-responsible-analytics-main"

# --- Helper Functions ---
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

check_requirements() {
    log "Checking requirements..."
    for cmd in curl unzip; do
        if ! command -v $cmd &> /dev/null; then
            log "ERROR: $cmd is required but not installed"
            exit 1
        fi
    done
}

check_disk_space() {
    local required_space=1000000  # 1GB in KB
    local available_space=$(df -k . | awk 'NR==2 {print $4}')
    
    if [ "$available_space" -lt "$required_space" ]; then
        log "ERROR: Insufficient disk space. Required: 1GB, Available: $((available_space/1000))MB"
        exit 1
    fi
}

# --- Script Main Logic ---


# Check requirements
check_requirements

# Check disk space
check_disk_space

# Download the repository
log "Downloading repository from GitHub..."
if ! curl -L --max-time 300 "$REPO_URL" -o repo.zip; then
    log "ERROR: Failed to download repository (timeout after 5 minutes)"
    exit 1
fi

# Extract the zip file
log "Extracting repository..."
if ! unzip -o repo.zip; then
    log "ERROR: Failed to extract repository"
    exit 1
fi

# Remove the zip file after extraction
rm repo.zip

# Move to the extracted directory
log "Moving to project directory..."
cd "$REPO_NAME"

# Make deploy script executable
log "Making deploy script executable..."
if ! chmod +x deploy.sh; then
    log "ERROR: Failed to make deploy script executable"
    exit 1
fi

# Run the deploy script with provided arguments
log "Running deployment script..."
./deploy.sh MobileNet chest_xray training

# Store the exit code
DEPLOY_EXIT_CODE=$?

# Exit with the same code as the deploy script
exit $DEPLOY_EXIT_CODE