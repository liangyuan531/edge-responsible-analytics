from prometheus_client import start_http_server, Gauge, Counter
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitoringService:
    def __init__(self, prometheus_port: int = 8000):
        self.model_performance = Gauge('model_performance', 'Model performance metrics', ['metric_name'])
        self.model_updates = Counter('model_updates', 'Number of model updates received')
        self.resource_usage = Gauge('resource_usage', 'Resource usage metrics', ['resource_type'])
        
        # Start Prometheus metrics server
        start_http_server(prometheus_port)
        
    def track_performance(self, metrics: Dict[str, float]):
        """
        Track model performance metrics
        """
        try:
            for metric_name, value in metrics.items():
                self.model_performance.labels(metric_name).set(value)
                logger.info(f"Tracked performance metric {metric_name}: {value}")
        except Exception as e:
            logger.error(f"Error tracking performance: {str(e)}")
            
    def track_resource_usage(self, cpu_usage: float, memory_usage: float, gpu_usage: float = None):
        """
        Track resource usage metrics
        """
        self.resource_usage.labels('cpu').set(cpu_usage)
        self.resource_usage.labels('memory').set(memory_usage)
        if gpu_usage is not None:
            self.resource_usage.labels('gpu').set(gpu_usage) 