import logging
import json
import paho.mqtt.client as mqtt
import base64
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TrainingSubmitter:
    def __init__(self, mqtt_client: mqtt.Client, device_id: str):
        self.mqtt_client = mqtt_client
        self.device_id = device_id
        self.training_topic = 'training/submit'
        self.status_topic = f'training/status/{device_id}'
        
    def submit_training_task(self, config: Dict[str, Any]) -> bool:
        """
        Submit a training task to the edge processing layer.
        """
        try:
            payload = {
                'device_id': self.device_id,
                'config': config
            }
            
            # Publish training task
            result = self.mqtt_client.publish(
                self.training_topic,
                json.dumps(payload)
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Training task submitted successfully")
                return True
            else:
                logger.error(f"Failed to submit training task: {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting training task: {e}")
            return False
            
    def on_training_status(self, client, userdata, message):
        """
        Handle training status updates from the edge processing layer.
        """
        try:
            status = json.loads(message.payload.decode())
            if status['device_id'] == self.device_id:
                logger.info(f"Training status update: {status['status']}")
                if status['status'] == 'completed':
                    self._handle_completed_training(status)
                elif status['status'] == 'failed':
                    logger.error(f"Training failed: {status.get('error')}")
                    
        except Exception as e:
            logger.error(f"Error processing training status: {e}")
            
    def _handle_completed_training(self, status: Dict[str, Any]):
        """
        Handle completed training status.
        """
        try:
            metrics = status.get('metrics', {})
            logger.info(f"Training completed with metrics: {metrics}")
            
            # Handle model deployment if evaluation passed
            evaluation = metrics.get('evaluation_metrics', {})
            if all(evaluation.get(policy, {}).get('passed', False) 
                  for policy in ['fairness', 'reliability', 'explainability']):
                logger.info("Model passed all evaluations, ready for deployment")
            else:
                logger.warning("Model failed some evaluations, deployment blocked")
                
        except Exception as e:
            logger.error(f"Error handling completed training: {e}") 