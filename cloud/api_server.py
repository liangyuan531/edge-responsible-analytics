from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from model_aggregation import FederatedModelAggregator
from monitoring_service import ModelMonitoringService
import logging
import base64
import json
import os
import tempfile
import shutil
import tarfile
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5002')
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', 8000))
MODEL_STORAGE_PATH = os.getenv('MODEL_STORAGE_PATH', '/app/models')

# Ensure model storage directory exists
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# Initialize services
aggregator = FederatedModelAggregator(mlflow_uri=MLFLOW_URI)
monitor = ModelMonitoringService(prometheus_port=PROMETHEUS_PORT)

@app.get("/")
async def root():
    return {"message": "Cloud Layer API"}

@app.get("/health")
async def health_check():
    """Health check endpoint for k8s probes"""
    try:
        # Check MLflow connection
        mlflow_status = "healthy" if aggregator.check_mlflow_connection() else "unhealthy"
        return {
            "status": "healthy",
            "mlflow": mlflow_status,
            "timestamp": str(datetime.now())
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ModelUpdate(BaseModel):
    edge_server_id: str
    model_params: str  # base64 encoded model
    metrics: Dict[str, float]
    model_type: str

@app.post("/edge/update")
async def receive_edge_update(update: ModelUpdate):
    """
    Receive model updates from edge processing servers
    """
    try:
        # Track the update
        monitor.track_performance(update.metrics)
        
        # Store the update in MLflow
        logger.info(f"Received update from edge server {update.edge_server_id}")
        
        # Trigger aggregation if we have enough updates
        aggregated_model = await aggregator.aggregate_models([update.model_params])
        
        # Evaluate aggregated model
        passed_policies, failed_policies = aggregator.evaluate_aggregated_model(
            model=aggregated_model,
            validation_data=update.metrics.get('validation_data'),
            thresholds=update.metrics.get('thresholds')
        )
        
        if passed_policies:
            try:
                # Encode the aggregated model
                if isinstance(aggregated_model, str):  # If path to saved model
                    if os.path.isdir(aggregated_model):
                        # Handle directory-based models
                        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                            shutil.make_archive(tmp.name[:-7], 'gztar', aggregated_model)
                            with open(tmp.name, 'rb') as f:
                                model_bytes = f.read()
                            os.unlink(tmp.name)
                    else:
                        # Handle single file models
                        with open(aggregated_model, 'rb') as f:
                            model_bytes = f.read()
                else:
                    # Handle model object (needs serialization)
                    model_bytes = aggregator.serialize_model(aggregated_model)
                
                model_b64 = base64.b64encode(model_bytes).decode('utf-8')
                
                return {
                    "status": "success",
                    "model": model_b64,
                    "model_type": update.model_type,
                    "message": "Model aggregated successfully"
                }
                
            except Exception as e:
                logger.exception(f"Error processing global model: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to process global model: {str(e)}"
                }
        else:
            return {
                "status": "failed",
                "failed_policies": failed_policies,
                "message": "Model failed policy validation"
            }
            
    except Exception as e:
        logger.error(f"Error processing update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Add error handling for the server startup
    try:
        logger.info(f"Starting cloud server with MLflow URI: {MLFLOW_URI}")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8080,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise 