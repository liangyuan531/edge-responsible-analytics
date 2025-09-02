import mlflow
from typing import List, Dict
import numpy as np
from datetime import datetime
import logging
import sys
import os
import tempfile
import base64

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from utils.policy_evaluator import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedModelAggregator:
    def __init__(self, mlflow_uri=None):
        self.mlflow_uri = mlflow_uri
        if mlflow_uri:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                logger.info(f"MLflow tracking URI set to {mlflow_uri}")
            except Exception as e:
                logger.warning(f"Failed to set MLflow tracking URI: {e}")

    def check_mlflow_connection(self):
        """Check if MLflow connection is working"""
        try:
            # Attempt to list experiments to verify connection
            mlflow.search_experiments()
            return True
        except Exception:
            return False

    async def aggregate_models(self, model_params_list):
        """
        Aggregate models from different edge servers.
        
        Args:
            model_params_list: List of base64 encoded model parameters
            
        Returns:
            Aggregated model or path to saved model
        """
        try:
            if not model_params_list:
                raise ValueError("No models to aggregate")
                
            # For now, with one model, just return the first one
            # In a real system, you would implement actual model aggregation here
            
            # Just pass through the first model for now
            # This is a temporary placeholder for actual federated averaging
            return model_params_list[0]
            
        except Exception as e:
            logger.error(f"Error during model aggregation: {e}")
            raise

    def evaluate_aggregated_model(self, model, validation_data=None, thresholds=None):
        """
        Evaluate if the aggregated model meets policy requirements.
        
        Returns:
            (bool, list): (passed_policies, failed_policies)
        """
        # Simplified implementation - in a real system you would evaluate the model
        # against your policy requirements
        
        # For now, assume model passes all policies
        return True, []
        
    def serialize_model(self, model):
        """
        Serialize model object to bytes.
        
        Args:
            model: Model object or string reference
            
        Returns:
            bytes: Serialized model
        """
        try:
            if isinstance(model, str):
                # If it's already a base64 string, decode it to bytes
                try:
                    return base64.b64decode(model)
                except Exception as e:
                    logger.error(f"Error decoding base64 string: {e}")
                    # If we can't decode as base64, check if it's a file path
                    if os.path.exists(model):
                        with open(model, 'rb') as f:
                            return f.read()
                    else:
                        # Last resort, just encode the string as bytes
                        return model.encode('utf-8')
            else:
                # Handle actual model object serialization
                import io
                import pickle
                
                buffer = io.BytesIO()
                pickle.dump(model, buffer)
                buffer.seek(0)
                return buffer.getvalue()
        except Exception as e:
            logger.exception(f"Error serializing model: {e}")
            raise