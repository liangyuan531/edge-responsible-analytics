import logging
import threading
from typing import Dict, Any
import json
import os
from .training_coordinator import TrainingCoordinator
from utils.policy_evaluator import (
    evaluate_fairness_policy,
    evaluate_reliability_policy,
    evaluate_explainability_policy
)
import tensorflow as tf
from utils.training import train_model

logger = logging.getLogger(__name__)

class TrainingHandler:
    def __init__(self, coordinator: TrainingCoordinator):
        self.coordinator = coordinator
        self.running = True
        self.thread = threading.Thread(target=self._process_training_tasks)
        self.thread.daemon = True

    def start(self):
        """
        Start the training handler thread.
        """
        self.thread.start()
        logger.info("Training handler started")

    def stop(self):
        """
        Stop the training handler thread.
        """
        self.running = False
        self.thread.join()
        logger.info("Training handler stopped")

    def _process_training_tasks(self):
        """
        Process training tasks from the queue.
        """
        while self.running:
            task = self.coordinator.get_next_task()
            if task:
                try:
                    device_id = task['device_id']
                    config = task['config']
                    
                    # Update status to training
                    self.coordinator.update_training_status(device_id, 'training')
                    
                    # Perform training
                    training_result = self._train_model(config)
                    
                    # Evaluate model
                    evaluation_result = self._evaluate_model(
                        training_result['model'],
                        training_result['validation_data']
                    )
                    
                    # Update status with results
                    self.coordinator.update_training_status(
                        device_id, 
                        'completed',
                        {
                            'training_metrics': training_result['metrics'],
                            'evaluation_metrics': evaluation_result
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing training task: {e}")
                    self.coordinator.update_training_status(device_id, 'failed')
                finally:
                    self.coordinator.cleanup_completed_task(device_id)

    def _train_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model based on configuration.
        """
        try:
            model_type = config.get('model_type')
            data_path = config.get('data_path')
            batch_size = config.get('batch_size', 16)
            epochs = config.get('epochs', 1)
            
            data_type = 'chest_xray' if model_type == 'MobileNet' else 'mt'
            result = train_model(
                data_path=data_path,
                data_type=data_type,
                batch_size=batch_size,
                epochs=epochs,
                model_type=model_type
            )
                
            return result
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def _evaluate_model(self, model, validation_data) -> Dict[str, Any]:
        """
        Evaluate the trained model using policy evaluators.
        """
        try:
            X_val, y_val, sensitive_features = validation_data
            
            # Evaluate fairness
            is_fair, failed_fairness = evaluate_fairness_policy(
                model, X_val, y_val, sensitive_features,
                thresholds={'demographic_parity_difference': 0.1}
            )
            
            # Evaluate reliability
            is_reliable, failed_reliability = evaluate_reliability_policy(
                model, X_val, y_val,
                thresholds={'reliability_score': 0.8}
            )
            
            # Evaluate explainability
            is_explainable, failed_explainability = evaluate_explainability_policy(
                model, X_val,
                thresholds={'explainability_score': 0.7}
            )
            
            return {
                'fairness': {
                    'passed': is_fair,
                    'failed_policies': failed_fairness
                },
                'reliability': {
                    'passed': is_reliable,
                    'failed_policies': failed_reliability
                },
                'explainability': {
                    'passed': is_explainable,
                    'failed_policies': failed_explainability
                }
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise 