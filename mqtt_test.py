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

# Callback when client receives a message
def on_message(client, userdata, msg):
    logger.info(f"Received message on topic: {msg.topic}")
    try:
        payload = msg.payload.decode()
        logger.info(f"Message payload: {payload}")
    except Exception as e:
        logger.error(f"Error decoding message: {e}")

# Callback when client connects
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Successfully connected to MQTT broker")
        # Subscribe to test topic and the actual topics we use
        topics = [
            (TEST_TOPIC, 1),
            ("models/aggregated/control", 1),
            ("models/aggregated/chunks", 1)
        ]
        client.subscribe(topics)
        logger.info(f"Subscribed to topics: {topics}")
    else:
        logger.error(f"Failed to connect to broker with result code: {rc}")

# Callback when client subscribes
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    logger.info(f"Subscribed with message ID: {mid}, QoS: {granted_qos}")

# Create MQTT client
client = mqtt.Client(client_id="test_subscriber", protocol=mqtt.MQTTv5)
client.on_connect = on_connect
client.on_message = on_message
client.on_subscribe = on_subscribe

# Connect to broker
try:
    logger.info(f"Connecting to broker at {MQTT_BROKER}:{MQTT_PORT}")
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()

    # Keep the script running
    while True:
        if client.is_connected():
            logger.info("Still connected and listening...")
        else:
            logger.warning("Connection lost!")
        time.sleep(5)

except KeyboardInterrupt:
    logger.info("Stopping...")
    client.loop_stop()
    client.disconnect()
except Exception as e:
    logger.error(f"Error: {e}") 