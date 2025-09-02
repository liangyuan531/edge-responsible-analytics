import base64
import json
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelTransfer:
    def __init__(self, cloud_api_url: str):
        self.cloud_api_url = cloud_api_url
        
    def upload_model_to_cloud(self, model_data: bytes, edge_server_id: str, model_type: str, metrics: Dict[str, float]) -> bool:
        """
        Upload a model from edge processing server to cloud
        """
        try:
            model_b64 = base64.b64encode(model_data).decode('utf-8')
            payload = {
                "edge_server_id": edge_server_id,
                "model_params": model_b64,
                "model_type": model_type,
                "metrics": metrics
            }
            
            response = requests.post(
                f"{self.cloud_api_url}/edge/update",
                json=payload
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully uploaded model from edge server {edge_server_id}")
                return True
            else:
                logger.error(f"Failed to upload model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            return False
            
    def download_aggregated_model(self) -> Dict[str, Any]:
        """
        Download the aggregated model from the cloud.
        """
        try:
            response = requests.get(f"{self.cloud_api_url}/model/aggregated")
            
            if response.status_code == 200:
                model_data = response.json()
                model_bytes = base64.b64decode(model_data["model_params"])
                return {
                    "model_data": model_bytes,
                    "model_type": model_data["model_type"]
                }
            else:
                logger.error(f"Failed to download aggregated model: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading aggregated model: {str(e)}")
            return None 