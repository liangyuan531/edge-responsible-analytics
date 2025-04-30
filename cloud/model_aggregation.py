import mlflow
from typing import List, Dict
import numpy as np
from datetime import datetime
import logging
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from utils.policy_evaluator import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedModelAggregator:
    def __init__(self, mlflow_uri: str):
        self.mlflow_uri = mlflow_uri
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"Connected to MLflow at {mlflow_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {e}")
            raise

    def check_mlflow_connection(self) -> bool:
        """Check if MLflow connection is healthy"""
        try:
            mlflow.search_experiments()
            return True
        except Exception as e:
            logger.error(f"MLflow connection check failed: {e}")
            return False

    def serialize_model(self, model):
        """Serialize model to bytes"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                np.save(tmp.name, model)
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
                os.unlink(tmp.name)
            return model_bytes
        except Exception as e:
            logger.error(f"Error serializing model: {e}")
            raise

    def aggregate_models(self, model_updates: List[Dict], weights: List[float] = None):
        """
        Aggregate model parameters from multiple edge devices
        """
        if weights is None:
            weights = [1/len(model_updates)] * len(model_updates)
            
        try:
            aggregated_params = {}
            for param_name in model_updates[0].keys():
                weighted_params = []
                for update, weight in zip(model_updates, weights):
                    weighted_params.append(update[param_name] * weight)
                aggregated_params[param_name] = np.sum(weighted_params, axis=0)
                
            # Log the aggregated model to MLflow
            with mlflow.start_run(run_name=f"aggregation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_params({"n_models": len(model_updates)})
                mlflow.pytorch.log_model(aggregated_params, "aggregated_model")
                
            return aggregated_params
            
        except Exception as e:
            logger.error(f"Error during model aggregation: {str(e)}")
            raise 

    def evaluate_aggregated_model(self, model, validation_data, thresholds):
        """
        Evaluate the aggregated model using all policies.
        """
        X_val, y_val, sensitive_features = validation_data
        
        # Evaluate all policies
        fairness_result = evaluate_fairness_policy(model, X_val, y_val, sensitive_features, thresholds.get('fairness'))
        reliability_result = evaluate_reliability_policy(model, X_val, y_val, thresholds.get('reliability'))
        explainability_result = evaluate_explainability_policy(model, X_val, thresholds.get('explainability'))
        
        # Combine results
        passed_all = all([
            fairness_result[0],
            reliability_result[0],
            explainability_result[0]
        ])
        
        failed_policies = (
            fairness_result[1] +
            reliability_result[1] +
            explainability_result[1]
        )
        
        return passed_all, failed_policies