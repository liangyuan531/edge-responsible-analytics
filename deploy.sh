#!/bin/bash

# Comprehensive script to deploy all services and configure experiments.
#
# Usage:
# ./deploy_all.sh <model_type> <data_type> <task_type>
# Example:
# ./deploy_all.sh MobileNet chest_xray training
# ./deploy_all.sh tinybert mt inference

# --- Configuration ---
NAMESPACE="default" # Change if your resources are in a different namespace

# Timeouts and Delays
EDGE_PROCESSING_READY_TIMEOUT_SECONDS=120 # 5 minutes to wait for edge-processing servers
EDGE_DEVICE_RELOAD_DELAY_SECONDS=60    # 2 minutes to wait for edge-device pods to be reloaded by Stakater Reloader

RELEASE_DIR="edge-responsible-analytics-release"

# Deployment Paths (relative to the script's location)
CONFIGMAPS_DIR="k3s/configmaps"
SERVICES_DIR="k3s/services"
STORAGE_DIR="k3s/storage"
DEPLOYMENTS_DIR="k3s/deployments"

# Specific deployment files
OPA_DEPLOYMENT="$DEPLOYMENTS_DIR/opa-deployment.yaml" 
MQTT_DEPLOYMENT1="$DEPLOYMENTS_DIR/mqtt-deployment1.yaml"
MQTT_DEPLOYMENT2="$DEPLOYMENTS_DIR/mqtt-deployment2.yaml"
MLFLOW_DEPLOYMENT="$DEPLOYMENTS_DIR/mlflow-deployment.yaml"
CLOUD_DEPLOYMENT="$DEPLOYMENTS_DIR/cloud-deployment.yaml"
EDGE_PROCESSING_DEPLOYMENT1="$DEPLOYMENTS_DIR/edge-processing-deployment1.yaml"
EDGE_PROCESSING_DEPLOYMENT2="$DEPLOYMENTS_DIR/edge-processing-deployment2.yaml"
EDGE_DEVICE_DEPLOYMENT1="$DEPLOYMENTS_DIR/edge-device-deployment1.yaml"
EDGE_DEVICE_DEPLOYMENT2="$DEPLOYMENTS_DIR/edge-device-deployment2.yaml"

CONFIGMAP_EDGE_DEVICE_CONFIG_NAME="edge-device-config"

# --- Helper Functions ---
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

apply_yaml_files_in_dir() {
  local dir_path="$1"
  local current_namespace="$2"
  log "Applying all YAML files in directory: $dir_path ..."
  if [ ! -d "$dir_path" ]; then
    log "ERROR: Directory $dir_path does not exist. Skipping."
    return 1
  fi

  find "$dir_path" -maxdepth 1 \( -name "*.yaml" -o -name "*.yml" \) -print0 | while IFS= read -r -d $'\0' file; do
    log "  Applying $file ..."
    sudo k3s kubectl apply -n "$current_namespace" -f "$file"
    if [ $? -ne 0 ]; then
      log "  ERROR: Failed to apply $file. Continuing..."
      # Consider adding 'exit 1' here if any failure should stop the script
    else
      log "  Successfully applied $file."
    fi
  done
  log "Finished applying YAML files in $dir_path."
}

apply_yaml_file() {
  local file_path="$1"
  local current_namespace="$2"
  local component_name="$3" # For logging
  log "Applying $component_name deployment ($file_path)..."
  if [ ! -f "$file_path" ]; then
    log "ERROR: File $file_path does not exist. Skipping."
    return 1
  fi
  kubectl apply -n "$current_namespace" -f "$file_path"
  if [ $? -ne 0 ]; then
    log "ERROR: Failed to apply $file_path for $component_name. Continuing..."
    # Consider adding 'exit 1' here
  else
    log "Successfully applied $file_path for $component_name."
  fi
}

wait_for_statefulset_ready() {
  local sts_name="$1"
  local expected_replicas="$2"
  local current_namespace="$3"
  local timeout_seconds="$4"
  local ready_replicas=0
  local start_time
  start_time=$(date +%s)

  log "Waiting up to $timeout_seconds seconds for StatefulSet '$sts_name' in namespace '$current_namespace' to have $expected_replicas ready replica(s)..."

  while true; do
    # Fetch current readyReplicas. If the STS doesn't exist yet or status isn't populated, this might be empty or error.
    ready_replicas_output=$(kubectl get statefulset "$sts_name" -n "$current_namespace" -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
    
    if [[ -n "$ready_replicas_output" && "$ready_replicas_output" =~ ^[0-9]+$ ]]; then
      ready_replicas=$ready_replicas_output
    else
      ready_replicas=0 # Default to 0 if not found or not a number
    fi

    if [[ "$ready_replicas" -ge "$expected_replicas" ]]; then
      log "StatefulSet '$sts_name' is ready with $ready_replicas out of $expected_replicas desired replica(s)."
      return 0
    fi

    local current_time
    current_time=$(date +%s)
    local elapsed_time=$((current_time - start_time))

    if [[ "$elapsed_time" -ge "$timeout_seconds" ]]; then
      log "ERROR: Timeout waiting for StatefulSet '$sts_name' to become ready. Last seen ready replicas: $ready_replicas."
      return 1 # Indicate failure
    fi

    log "StatefulSet '$sts_name': $ready_replicas/$expected_replicas ready. Waiting..."
    sleep 15 # Check every 15 seconds
  done
}

# --- Script Main Logic ---

# Save all environment variables to a log file with a timestamp
printenv > "env_log_$(date +%Y%m%d_%H%M%S).txt"

# Check for correct number of arguments for experiment configuration
if [ "$#" -ne 3 ]; then
  log "ERROR: Incorrect number of arguments."
  log "Usage: $0 <model_type> <data_type> <task_type>"
  log "Example: $0 MobileNet chest_xray training"
  exit 1
fi

MODEL_TYPE="$1"
DATA_TYPE="$2"
TASK_TYPE="$3"

log "--- Starting Full Deployment and Experiment Setup ---"
log "Target Namespace: $NAMESPACE"
log "Experiment Params: Model=$MODEL_TYPE, Data=$DATA_TYPE, Task=$TASK_TYPE"

# Phase 1: Initial Setup (ConfigMaps, Services, Storage)
log ""
log "--- Phase 1: Applying Initial Configurations ---"
apply_yaml_files_in_dir "$CONFIGMAPS_DIR" "$NAMESPACE"
apply_yaml_files_in_dir "$SERVICES_DIR" "$NAMESPACE"
apply_yaml_files_in_dir "$STORAGE_DIR" "$NAMESPACE"

# Phase 2: Core Application Deployments
log ""
log "--- Phase 2: Deploying Core Applications ---"
apply_yaml_file "$OPA_DEPLOYMENT" "$NAMESPACE" "OPA"
apply_yaml_file "$MQTT_DEPLOYMENT1" "$NAMESPACE" "MQTT Server 1"
apply_yaml_file "$MQTT_DEPLOYMENT2" "$NAMESPACE" "MQTT Server 2"
apply_yaml_file "$MLFLOW_DEPLOYMENT" "$NAMESPACE" "MLflow"
apply_yaml_file "$CLOUD_DEPLOYMENT" "$NAMESPACE" "Cloud Layer"
apply_yaml_file "$EDGE_PROCESSING_DEPLOYMENT1" "$NAMESPACE" "Edge Processing Server 1"
apply_yaml_file "$EDGE_PROCESSING_DEPLOYMENT2" "$NAMESPACE" "Edge Processing Server 2"

# Phase 3: Wait for Edge Processing Servers to be Ready
log ""
log "--- Phase 3: Waiting for Edge Processing Servers ---"
wait_for_statefulset_ready "edge-processing-server-1" 1 "$NAMESPACE" "$EDGE_PROCESSING_READY_TIMEOUT_SECONDS"
if [ $? -ne 0 ]; then log "Warning: edge-processing-server-1 may not be fully ready."; fi

wait_for_statefulset_ready "edge-processing-server-2" 1 "$NAMESPACE" "$EDGE_PROCESSING_READY_TIMEOUT_SECONDS"
if [ $? -ne 0 ]; then log "Warning: edge-processing-server-2 may not be fully ready."; fi
sleep "$EDGE_PROCESSING_READY_TIMEOUT_SECONDS"

# Phase 4: Configure and Deploy Edge Devices for the experiment
log ""
log "--- Phase 4: Configuring and Deploying Edge Devices for Experiment ---"
log "Updating ConfigMap '$CONFIGMAP_EDGE_DEVICE_CONFIG_NAME' for the experiment..."
kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: $CONFIGMAP_EDGE_DEVICE_CONFIG_NAME
  namespace: $NAMESPACE
data:
  model_type: "$MODEL_TYPE"
  data_type: "$DATA_TYPE"
  task_type: "$TASK_TYPE"
EOF

if [ $? -ne 0 ]; then
  log "ERROR: Failed to apply/update ConfigMap '$CONFIGMAP_EDGE_DEVICE_CONFIG_NAME'."
  exit 1
fi
log "ConfigMap '$CONFIGMAP_EDGE_DEVICE_CONFIG_NAME' updated successfully for the experiment."

apply_yaml_file "$EDGE_DEVICE_DEPLOYMENT1" "$NAMESPACE" "Edge Device Domain 1"
apply_yaml_file "$EDGE_DEVICE_DEPLOYMENT2" "$NAMESPACE" "Edge Device Domain 2"

log "Waiting $EDGE_DEVICE_RELOAD_DELAY_SECONDS seconds for edge-device pods to be (potentially) restarted by Reloader due to ConfigMap change..."
sleep "$EDGE_DEVICE_RELOAD_DELAY_SECONDS"
log "Wait for potential edge-device reload complete."

log ""
log "--- Full Deployment Script Finished ---"
log "Edge processing servers are configured to dynamically read experiment settings."
log "Edge device pods should have picked up the latest experiment configuration via Reloader."
log "Please monitor your Kubernetes dashboard and pod logs to ensure everything is running as expected."
