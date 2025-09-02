import tensorflow as tf
import numpy as np
import time
from load_models import load_mobilenet_model, load_bert_model
import pandas as pd
import logging
from typing import Generator, Tuple, Union, List
import gc

logger = logging.getLogger(__name__)

def preprocess_image(image):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

def perform_inference(data, data_type, batch_size=16):
    try:
        logger.info(f"[edge_infer] Starting inference on {data}")
        if data_type in ["chest_xray", "cxr8"]:
            model = load_mobilenet_model()
            # Handle generator input (for CXR8)
            if isinstance(data, Generator):
                predictions = []
                sensitive_features = []

                logger.info(f"Starting inference on {data_type} data, model loaded {model}")

                for batch_data in data:
                    if isinstance(batch_data, tuple):
                        X_batch, _, sf_batch = batch_data
                        batch_predictions = model.predict(X_batch, batch_size=batch_size)
                        predictions.extend(batch_predictions.flatten().tolist())
                        # Ensure sf_batch is a DataFrame before appending
                        if isinstance(sf_batch, pd.DataFrame):
                            sensitive_features.append(sf_batch)
                        elif isinstance(sf_batch, np.ndarray):
                            # Convert numpy array to DataFrame if necessary
                            sensitive_features.append(pd.DataFrame({
                                'gender': sf_batch if sf_batch.ndim == 1 else sf_batch[:, 0],
                                'age_group': sf_batch[:, 1] if sf_batch.ndim > 1 else None
                            }))
                
                logger.info(f"Inference completed for {data_type} data")
                # Return predictions along with sensitive features for evaluation
                return {
                    'predictions': np.array(predictions),
                    'sensitive_features': pd.concat(sensitive_features, ignore_index=True) if sensitive_features else None
                }
                
            # Handle direct input (for chest_xray)
            else:
                predictions = model.predict(data)
                return predictions
            
        elif data_type == "mt" or data_type == "mimic":
            # For MIMIC, we'll use TinyBERT by default
            model, tokenizer = load_bert_model()
            
            # Handle generator input (for MIMIC)
            if isinstance(data, Generator):
                predictions = []
                sensitive_features = []
                
                logger.info(f"Starting inference on {data_type} data with TinyBERT")
                
                for batch_data in data:
                    if isinstance(batch_data, tuple):
                        texts, sf_batch = batch_data
                        # Tokenize the texts
                        inputs = tokenizer(
                            texts.tolist(),
                            padding=True,
                            truncation=True,
                            return_tensors="tf",
                            max_length=64
                        )
                        
                        # Get predictions
                        outputs = model(inputs)
                        batch_predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()
                        predictions.extend(batch_predictions.tolist())
                        
                        if isinstance(sf_batch, pd.DataFrame):
                            sensitive_features.append(sf_batch)
                
                logger.info(f"Inference completed for {data_type} data")
                return {
                    'predictions': np.array(predictions),
                    'sensitive_features': pd.concat(sensitive_features, ignore_index=True) if sensitive_features else None
                }
            
            # Handle direct input
            else:
                # Ensure data is in the correct format (list of strings)
                if isinstance(data, pd.Series):
                    data = data.tolist()
                elif isinstance(data, str):
                    data = [data]
                elif not isinstance(data, list):
                    raise ValueError(f"Unexpected data type for text data: {type(data)}")
                
                # Process in batches
                predictions = []
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    try:
                        # Tokenize with padding and truncation
                        inputs = tokenizer(
                            batch, 
                            return_tensors="tf", 
                            padding=True, 
                            truncation=True,
                            max_length=64
                        )
                        
                        # Get predictions
                        outputs = model(inputs)
                        logits = outputs.logits
                        predictions.extend(tf.nn.softmax(logits, axis=-1).numpy().tolist())
                        
                        # Force garbage collection after each batch
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Error during inference batch {i}: {str(e)}")
                        raise
                        
                return predictions
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

def process_inference_results(predictions: Union[np.ndarray, dict], data_type: str) -> dict:
    """
    Process inference results based on data type.
    
    Args:
        predictions: Raw predictions from model
        data_type: Type of data ('chest_xray', 'cxr8', or 'mt')
    
    Returns:
        Dictionary containing processed results
    """
    try:
        if data_type == "cxr8":
            if isinstance(predictions, dict):
                return {
                    'predictions': predictions['predictions'].tolist(),
                    'sensitive_features': predictions['sensitive_features'].to_dict('records')
                }
        
        # For other data types, return predictions directly
        return {'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions}
    
    except Exception as e:
        logger.error(f"Error processing inference results: {str(e)}")
        raise
