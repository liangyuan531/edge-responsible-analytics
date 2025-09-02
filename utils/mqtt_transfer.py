import base64
import json
import math
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE = 100 * 1024  # n KB chunks
MQTT_QOS = 1  # Use QoS 1 for reliable delivery

class ChunkedMQTTTransfer:
    def __init__(self, mqtt_client, device_id: str):
        self.mqtt_client = mqtt_client
        self.device_id = device_id
        self.received_chunks: Dict[str, Dict[int, str]] = {}
        self.total_chunks: Dict[str, int] = {}
        self.transfer_metadata: Dict[str, Dict[str, Any]] = {}  # Store metadata for each transfer
        self.transfer_progress: Dict[str, Dict[str, Any]] = {}  # Track progress for each transfer
        self.last_progress_log: Dict[str, float] = {}  # Track last progress log time
        
    def send_file_in_chunks(self, file_data: bytes, topic: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a large file in chunks over MQTT.
        
        Args:
            file_data: The file data to send
            topic: The MQTT topic to publish to
            metadata: Additional metadata to include with the file
        """
        try:
            total_chunks = math.ceil(len(file_data) / CHUNK_SIZE)
            transfer_id = f"{self.device_id}_{hash(file_data)}"
            
            # Initialize progress tracking
            self.transfer_progress[transfer_id] = {
                'total_chunks': total_chunks,
                'sent_chunks': 0,
                'start_time': time.time()
            }
            
            logger.info(f"Starting transfer {transfer_id} with {total_chunks} chunks in topic: {topic}")

            # Send transfer start message with metadata
            start_payload = {
                'type': 'transfer_start',
                'transfer_id': transfer_id,
                'total_chunks': total_chunks,
                'device_id': self.device_id,
                'metadata': metadata or {}
            }
            start_result = self.mqtt_client.publish(f"{topic}/control", json.dumps(start_payload), qos=MQTT_QOS)
            start_result.wait_for_publish()
            logger.info(f"Transfer start message published for {transfer_id}")
            
            # Add small delay after start message to ensure receiver is ready
            time.sleep(0.1)
            
            # Send chunks with progress tracking
            for chunk_num in range(total_chunks):
                start_idx = chunk_num * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, len(file_data))
                chunk_data = file_data[start_idx:end_idx]
                
                chunk_payload = {
                    'type': 'chunk',
                    'transfer_id': transfer_id,
                    'chunk_num': chunk_num,
                    'total_chunks': total_chunks,
                    'data': base64.b64encode(chunk_data).decode('utf-8'),
                    'device_id': self.device_id
                }
                
                # Publish with QoS 1 and wait for confirmation
                result = self.mqtt_client.publish(f"{topic}/chunks", json.dumps(chunk_payload), qos=MQTT_QOS)
                result.wait_for_publish()
                
                # Update progress
                self.transfer_progress[transfer_id]['sent_chunks'] = chunk_num + 1
                
                # Log progress every 5% or 30 seconds
                current_time = time.time()
                last_log_time = self.last_progress_log.get(transfer_id, 0)
                if (current_time - last_log_time >= 30) or (chunk_num % max(1, total_chunks // 20) == 0):
                    progress = (chunk_num + 1) / total_chunks * 100
                    elapsed = current_time - self.transfer_progress[transfer_id]['start_time']
                    rate = (chunk_num + 1) / elapsed if elapsed > 0 else 0
                    logger.info(f"Transfer {transfer_id}: {progress:.1f}% complete ({chunk_num + 1}/{total_chunks} chunks), "
                              f"Rate: {rate:.1f} chunks/sec")
                    self.last_progress_log[transfer_id] = current_time
                
                # Add small delay between chunks to prevent overwhelming the broker
                if chunk_num % 10 == 0:  # Adjust delay frequency
                    time.sleep(0.01)
            
            # Send transfer complete message
            complete_payload = {
                'type': 'transfer_complete',
                'transfer_id': transfer_id,
                'device_id': self.device_id
            }
            complete_result = self.mqtt_client.publish(f"{topic}/control", json.dumps(complete_payload), qos=MQTT_QOS)
            complete_result.wait_for_publish()
            logger.info(f"Transfer complete message published for {transfer_id}")
            
            # Cleanup progress tracking
            if transfer_id in self.transfer_progress:
                del self.transfer_progress[transfer_id]
            if transfer_id in self.last_progress_log:
                del self.last_progress_log[transfer_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending file in chunks: {e}")
            return False
            
    def handle_chunk_message(self, msg) -> Optional[Dict[str, Any]]:
        """
        Handle incoming chunk messages and reassemble the file when complete.
        Returns the complete file data and metadata when all chunks are received.
        """
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            msg_type = payload.get('type')
            transfer_id = payload.get('transfer_id')
            device_id = payload.get('device_id')

            if msg_type == 'transfer_start':
                self.received_chunks[transfer_id] = {}
                self.total_chunks[transfer_id] = payload['total_chunks']
                # Store metadata from transfer_start message
                self.transfer_metadata[transfer_id] = payload.get('metadata', {})
                logger.info(f"Started new transfer {transfer_id} with metadata: {self.transfer_metadata[transfer_id]}")
                self.transfer_progress[transfer_id] = {
                    'total_chunks': payload['total_chunks'],
                    'received_chunks': 0,
                    'start_time': time.time()
                }
                self.last_progress_log[transfer_id] = time.time()
                return None
                
            elif msg_type == 'chunk':
                if transfer_id not in self.received_chunks:
                    logger.warning(f"Received chunk for unknown transfer {transfer_id}")
                    return None
                    
                chunk_num = payload['chunk_num']
                chunk_data = base64.b64decode(payload['data'])
                self.received_chunks[transfer_id][chunk_num] = chunk_data
                
                # Update progress tracking
                if transfer_id in self.transfer_progress:
                    self.transfer_progress[transfer_id]['received_chunks'] += 1
                    current_time = time.time()
                    last_log_time = self.last_progress_log.get(transfer_id, 0)
                    
                    # Log progress every 5% or 30 seconds
                    if (current_time - last_log_time >= 30) or (chunk_num % max(1, self.total_chunks[transfer_id] // 20) == 0):
                        progress = len(self.received_chunks[transfer_id]) / self.total_chunks[transfer_id] * 100
                        elapsed = current_time - self.transfer_progress[transfer_id]['start_time']
                        rate = len(self.received_chunks[transfer_id]) / elapsed if elapsed > 0 else 0
                        logger.info(f"Transfer {transfer_id}: {progress:.1f}% complete "
                                  f"({len(self.received_chunks[transfer_id])}/{self.total_chunks[transfer_id]} chunks), "
                                  f"Rate: {rate:.1f} chunks/sec")
                        self.last_progress_log[transfer_id] = current_time

                # Check if we have all chunks
                if len(self.received_chunks[transfer_id]) == self.total_chunks[transfer_id]:
                    logger.info(f"Received all chunks from {transfer_id}")
                    # Reassemble the file
                    chunks = [self.received_chunks[transfer_id][i] 
                            for i in range(self.total_chunks[transfer_id])]
                    complete_data = b''.join(chunks)
                    
                    # Get metadata from stored transfer metadata
                    metadata = self.transfer_metadata.get(transfer_id, {})
                    
                    # Clean up
                    del self.received_chunks[transfer_id]
                    del self.total_chunks[transfer_id]
                    if transfer_id in self.transfer_metadata:
                        del self.transfer_metadata[transfer_id]
                    
                    return {
                        'data': complete_data,
                        'metadata': metadata,
                        'device_id': device_id
                    }
            
            elif msg_type == 'transfer_complete':
                # Cleanup progress tracking
                if transfer_id in self.transfer_progress:
                    del self.transfer_progress[transfer_id]
                if transfer_id in self.last_progress_log:
                    del self.last_progress_log[transfer_id]
                # Clean up if we somehow missed completing the transfer
                if transfer_id in self.received_chunks:
                    del self.received_chunks[transfer_id]
                if transfer_id in self.total_chunks:
                    del self.total_chunks[transfer_id]
                if transfer_id in self.transfer_metadata:
                    del self.transfer_metadata[transfer_id]
                    
            return None
            
        except Exception as e:
            logger.error(f"Error handling chunk message: {e}")
            logger.exception("Detailed error:")
            return None 