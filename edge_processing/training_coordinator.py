import logging
import threading
import queue
from typing import Dict, Any
import json
import os

logger = logging.getLogger(__name__)

class TrainingCoordinator:
    def __init__(self):
        self.training_queue = queue.Queue()
        self.active_trainings: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
    def submit_training_task(self, device_id: str, task_config: dict):
        """
        Submit a training task from an edge device.
        """
        try:
            logger.info(f"Received training task from device {device_id}")
            task = {
                'device_id': device_id,
                'config': task_config,
                'status': 'pending'
            }
            self.training_queue.put(task)
            with self.lock:
                self.active_trainings[device_id] = task
            logger.info(f"Training task queued for device {device_id}")
            return True
        except Exception as e:
            logger.error(f"Error submitting training task: {e}")
            return False

    def get_training_status(self, device_id: str) -> dict:
        """
        Get the status of a training task for a specific device.
        """
        with self.lock:
            return self.active_trainings.get(device_id, {}).get('status', 'not_found')

    def update_training_status(self, device_id: str, status: str, metrics: dict = None):
        """
        Update the status and metrics of a training task.
        """
        with self.lock:
            if device_id in self.active_trainings:
                self.active_trainings[device_id]['status'] = status
                if metrics:
                    self.active_trainings[device_id]['metrics'] = metrics
                logger.info(f"Updated training status for device {device_id}: {status}")

    def get_next_task(self) -> dict:
        """
        Get the next training task from the queue.
        """
        try:
            return self.training_queue.get_nowait()
        except queue.Empty:
            return None

    def cleanup_completed_task(self, device_id: str):
        """
        Clean up completed training task.
        """
        with self.lock:
            if device_id in self.active_trainings:
                del self.active_trainings[device_id] 