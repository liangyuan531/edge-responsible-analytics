import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import tensorflow as tf
import time
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from edge.load_models import load_mobilenet_model, load_t5_model, load_bert_model
from dataset.chest_xray_processor import process_chest_xray_data
from dataset.mt_processor import process_medical_transcriptions_data

def train_model(data_path, data_type, batch_size=16, epochs=1, model_type='MobileNet'):
    """
    Train a model based on data type and configuration.
    
    Args:
        data_path (str): Path to training data
        data_type (str): Type of data ('chest_xray' or 'mt')
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        model_type (str): Type of model to train ('MobileNet', 't5_small', or 'tinybert')
        
    Returns:
        dict: Training results including metrics and model
    """
    start_time = time.time()
    validation_data = None
    
    if data_type == 'chest_xray':
        model = load_mobilenet_model()
        train_data = []
        val_data = []
        
        # Process training data
        for X_batch, y_batch, sf_batch in process_chest_xray_data(data_path, batch_size):
            train_data.append((X_batch, y_batch, sf_batch))
            
        # Process validation data
        val_path = os.path.join(os.path.dirname(data_path), 'val')
        for X_val, y_val, sf_val in process_chest_xray_data(val_path, batch_size):
            val_data.append((X_val, y_val, sf_val))
            
        # Train the model
        history = {'loss': [], 'accuracy': []}
        for epoch in range(epochs):
            for X_batch, y_batch, _ in train_data:
                metrics = model.fit(X_batch, y_batch, epochs=1, verbose=1)
                history['loss'].extend(metrics.history['loss'])
                history['accuracy'].extend(metrics.history['accuracy'])
        
        # Save validation data for evaluation
        validation_data = (
            np.concatenate([x[0] for x in val_data]),
            np.concatenate([x[1] for x in val_data]),
            np.concatenate([x[2] for x in val_data])
        )
        
        model.save("mobilenet_model.keras")
        training_metrics = {
            "loss": np.mean(history['loss']),
            "accuracy": np.mean(history['accuracy'])
        }
        
    elif data_type == 'mt':
        if model_type == 't5_small':
            model, tokenizer = load_t5_model()
        elif model_type == 'tinybert':
            model, tokenizer = load_bert_model()
        else:
            raise ValueError(f"Unsupported model type for MT data: {model_type}")
            
        # Process data
        X_train, X_test, y_train, y_test, sf_train, sf_test = process_medical_transcriptions_data(data_path)
        
        # Tokenize inputs
        if model_type == 't5_small':
            inputs = tokenizer(X_train.tolist(), return_tensors="tf", padding=True, truncation=True)
            labels = tokenizer(y_train.tolist(), return_tensors="tf", padding=True, truncation=True).input_ids
            dataset = tf.data.Dataset.from_tensor_slices((inputs.input_ids, labels)).shuffle(buffer_size=1024).batch(batch_size)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            
        elif model_type == 'tinybert':
            # Prepare data for TinyBERT
            inputs = tokenizer(
                X_train.tolist(),
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Convert labels to numerical format
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            
            # Create TensorFlow dataset
            dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'token_type_ids': inputs['token_type_ids']
                },
                y_train_encoded
            )).shuffle(1024).batch(batch_size)
            
            # Prepare validation data
            val_inputs = tokenizer(
                X_test.tolist(),
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=512
            )
            y_test_encoded = label_encoder.transform(y_test)
            
            validation_data = (
                {
                    'input_ids': val_inputs['input_ids'],
                    'attention_mask': val_inputs['attention_mask'],
                    'token_type_ids': val_inputs['token_type_ids']
                },
                y_test_encoded,
                sf_test
            )
            
            # Configure training
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = ['accuracy']
        
        # Compile and train
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        history = model.fit(dataset, epochs=epochs, verbose=1)
        
        # Save model
        if model_type == 't5_small':
            model.save_pretrained("t5_model")
        else:
            model.save_pretrained("tinybert_model")
            
        training_metrics = {
            "loss": float(history.history['loss'][-1]),
            "accuracy": float(history.history.get('accuracy', [0])[-1])
        }
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "model": model,
        "metrics": {
            **training_metrics,
            "duration": duration
        },
        "validation_data": validation_data
    }
