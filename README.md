# Edge-responsible-analytics
 
## 1. Edge Devices (Raspberry Pis): Perform local inference and send models to Edge Processing Servers via MQTT.

## 2. Edge Processing Servers: Aggregate models from multiple Edge Devices, evaluate them using OPA (Open Policy Agent), and periodically send aggregated models to the Cloud Server.

## 3. Cloud Server: Further aggregates models from multiple Edge Processing Servers, evaluates them using OPA, ensures high model performance, and manages model deployment.

### Kubernetes will orchestrate the services across these layers, ensuring seamless communication, scalability, and manageability.