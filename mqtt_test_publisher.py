import paho.mqtt.client as mqtt
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MQTT Configuration
MQTT_BROKER = '10.200.3.159'
MQTT_PORT = 1883
TEST_TOPIC = "test/message"

# Callback when client connects
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Successfully connected to MQTT broker")
    else:
        logger.error(f"Failed to connect to broker with result code: {rc}")

# Create MQTT client
client = mqtt.Client(client_id="test_publisher", protocol=mqtt.MQTTv5)
client.on_connect = on_connect

# Connect to broker
try:
    logger.info(f"Connecting to broker at {MQTT_BROKER}:{MQTT_PORT}")
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()

    # Wait for connection
    time.sleep(1)

    # Send test messages
    for i in range(5):
        message = {
            "type": "test_message",
            "sequence": i,
            "timestamp": time.time()
        }
        
        # Publish to test topic
        client.publish(TEST_TOPIC, json.dumps(message), qos=1)
        logger.info(f"Published test message {i}")
        
        # Also publish to the actual topic we use
        client.publish("models/aggregated/control", json.dumps({
            "type": "transfer_start",
            "transfer_id": f"test_{i}",
            "total_chunks": 5,
            "device_id": "test_publisher"
        }), qos=1)
        logger.info(f"Published test control message {i}")
        
        time.sleep(2)

    logger.info("Finished sending test messages")
    time.sleep(2)  # Wait for messages to be delivered
    
except KeyboardInterrupt:
    logger.info("Stopping...")
except Exception as e:
    logger.error(f"Error: {e}")
finally:
    client.loop_stop()
    client.disconnect()
    logger.info("Disconnected from broker") 