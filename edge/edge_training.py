import os
import gc
import tensorflow as tf
import numpy as np
from dataset.chest_xray_processor import process_chest_xray_data
from dataset.mt_processor import process_medical_transcriptions_data
from edge.load_models import load_mobilenet_model, load_t5_model, load_bert_model, medical_specialties

# Define the training step function outside any loops
@tf.function(reduce_retracing=True)
def train_step(model, optimizer, inputs, targets, loss_fn):
    """
    Single training step function defined outside the loop
    with reduce_retracing enabled
    """
    with tf.GradientTape() as tape:
        # For BERT models, we need to handle the output structure
        outputs = model(inputs, training=True)
        if hasattr(outputs, 'logits'):
            predictions = outputs.logits
        else:
            predictions = outputs
        
        loss = loss_fn(targets, predictions)
    
    # Clean variable names to avoid optimizer issues
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    
    # Clean gradients of any None values
    gradients = [
        tf.zeros_like(var) if grad is None else grad
        for grad, var in zip(gradients, trainable_vars)
    ]
    
    # Clip gradients
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss, predictions

def train_mobilenet_edge(data_path, epochs=2, samples_per_class=50):
    """
    Train MobileNet with optimized memory usage and reduced retracing.
    Handles both directory-based datasets and HuggingFace datasets.
    """
    # Create the model
    model = load_mobilenet_model()
    batch_size = 8
    
    # Configure training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    
    # Training metrics
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # Determine if using HuggingFace dataset (CXR8) or directory-based dataset
    is_huggingface = data_path is None
    
    if is_huggingface:
        # Load CXR8 data using the processor
        from dataset.cxr8_processor import process_cxr8_data
        train_gen, val_gen = process_cxr8_data(batch_size=batch_size)
        
        # Training loop for HuggingFace dataset
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            steps = 0
            
            # Training steps
            for batch_data in train_gen:
                if isinstance(batch_data, tuple):
                    X_batch, y_batch, _ = batch_data
                    # Call train_step with all required arguments
                    loss, predictions = train_step(
                        model=model,
                        optimizer=optimizer,
                        inputs=X_batch,
                        targets=y_batch,
                        loss_fn=loss_fn
                    )
                    
                    train_acc_metric.update_state(y_batch, predictions)
                    
                    total_loss += loss
                    steps += 1
                    
                    if steps % 10 == 0:
                        print(f"Step {steps}, Loss: {total_loss/steps}")
                    
                    if steps * batch_size >= samples_per_class * 2:
                        break
            
            # Validation steps
            val_loss = 0
            val_steps = 0
            for val_batch in val_gen:
                if isinstance(val_batch, tuple):
                    val_images, val_labels, _ = val_batch
                    val_predictions = model(val_images, training=False)
                    val_loss += loss_fn(val_labels, val_predictions)
                    val_acc_metric.update_state(val_labels, val_predictions)
                    val_steps += 1
                    
                    if val_steps * batch_size >= samples_per_class:
                        break
            
            # Calculate epoch metrics
            epoch_loss = total_loss / steps
            epoch_accuracy = train_acc_metric.result().numpy()
            epoch_val_loss = val_loss / val_steps
            epoch_val_accuracy = val_acc_metric.result().numpy()
            
            # Update best metrics
            if epoch_val_accuracy > best_accuracy:
                best_accuracy = epoch_val_accuracy
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
            
            print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}, "
                  f"val_loss={epoch_val_loss:.4f}, val_accuracy={epoch_val_accuracy:.4f}")
            
            # Reset metrics
            train_acc_metric.reset_state()
            val_acc_metric.reset_state()
            gc.collect()
    else:
        # Original directory-based training code
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Use flow_from_directory with fixed image size
        train_generator = datagen.flow_from_directory(
            data_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        val_generator = datagen.flow_from_directory(
            data_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        # Training loop for directory-based dataset
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            steps = 0
            
            # Training steps
            for images, labels in train_generator:
                # Call train_step with all required arguments
                loss, predictions = train_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=images,
                    targets=labels,
                    loss_fn=loss_fn
                )
                
                train_acc_metric.update_state(labels, predictions)
                
                total_loss += loss
                steps += 1
                
                if steps % 10 == 0:
                    print(f"Step {steps}, Loss: {total_loss/steps}")
                
                if steps * batch_size >= samples_per_class * 2:
                    break
            
            # Validation steps
            val_loss = 0
            val_steps = 0
            for val_images, val_labels in val_generator:
                val_predictions = model(val_images, training=False)
                val_loss += loss_fn(val_labels, val_predictions)
                val_acc_metric.update_state(val_labels, val_predictions)
                val_steps += 1
                
                if val_steps * batch_size >= samples_per_class:
                    break
            
            # Calculate epoch metrics
            epoch_loss = total_loss / steps
            epoch_accuracy = train_acc_metric.result().numpy()
            epoch_val_loss = val_loss / val_steps
            epoch_val_accuracy = val_acc_metric.result().numpy()
            
            # Update best metrics
            if epoch_val_accuracy > best_accuracy:
                best_accuracy = epoch_val_accuracy
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
            
            print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}, "
                  f"val_loss={epoch_val_loss:.4f}, val_accuracy={epoch_val_accuracy:.4f}")
            
            # Reset metrics
            train_acc_metric.reset_state()
            val_acc_metric.reset_state()
            gc.collect()
    
    # Save the model
    model.save('mobilenet_model.keras')
    
    return {
        'loss': float(epoch_loss),
        'accuracy': float(epoch_accuracy),
        'val_loss': float(epoch_val_loss),
        'val_accuracy': float(epoch_val_accuracy),
        'best_accuracy': float(best_accuracy),
        'best_loss': float(best_loss)
    }

def train_t5_edge(data_path, epochs=5, max_samples=200):
    print("Starting T5 training on edge device...")
    
    # Load model and tokenizer
    model, tokenizer = load_t5_model()
    
    # Determine dataset type and load appropriate data
    if "mimic" in str(data_path).lower():
        from dataset.mimic_processor import process_mimic_data
        train_gen, _ = process_mimic_data(
            batch_size=4,
            max_samples=max_samples
        )
        is_mimic = True
    else:
        # Get limited dataset for MT
        X_train, _, y_train, _, _, _ = process_medical_transcriptions_data(
            data_path, batch_size=4
        )
        is_mimic = False
        
        # Limit dataset size
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
    
    # Configure optimizer with low learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # Training metrics
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # Training loop with reduced retracing
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        if is_mimic:
            # Training on MIMIC data using generator
            for texts, sensitive_features in train_gen:
                # Pre-tokenize batch
                inputs = tokenizer(
                    texts.tolist(),
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="tf"
                )
                
                # For MIMIC, we'll predict gender as a binary classification task
                targets = tf.convert_to_tensor(
                    sensitive_features['gender'].map({'male': 0, 'female': 1}).values,
                    dtype=tf.int32
                )
                
                # Train step
                loss, logits = train_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=loss_fn
                )
                
                # Update metrics
                train_acc_metric.update_state(targets, logits)
                
                total_loss += loss
                steps += 1
                
                if steps % 10 == 0:
                    accuracy = train_acc_metric.result().numpy()
                    print(f"Step {steps}, Loss: {total_loss/steps}, Accuracy: {accuracy}")
                
                # Force garbage collection
                gc.collect()
        else:
            # Original MT training code
            # Pre-tokenize all data at once
            inputs = tokenizer(
                list(X_train),
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="tf"
            )
            
            targets = tokenizer(
                list(y_train),
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="tf"
            )
            
            # Convert to tf.data.Dataset with fixed batch size
            batch_size = 2
            dataset = tf.data.Dataset.from_tensor_slices((
                {k: v for k, v in inputs.items()},
                targets.input_ids
            )).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            
            for batch_inputs, batch_targets in dataset:
                loss = train_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=batch_inputs,
                    targets=batch_targets,
                    loss_fn=loss_fn
                )
                
                # Update metrics
                train_acc_metric.update_state(batch_targets, loss)
                
                total_loss += loss
                steps += 1
                
                if steps % 10 == 0:
                    print(f"Step {steps}, Loss: {total_loss/steps}")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / steps
        epoch_accuracy = train_acc_metric.result().numpy()
        
        # Update best metrics
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}")
        
        # Reset metrics
        train_acc_metric.reset_state()
        gc.collect()
    
    # Save the model
    model.save_pretrained("t5_small")
    tokenizer.save_pretrained("t5_small")
    
    return {
        'loss': float(epoch_loss),
        'accuracy': float(epoch_accuracy),
        'best_accuracy': float(best_accuracy),
        'best_loss': float(best_loss)
    }

def train_bert_edge(data_path, epochs=5, max_samples=300):
    """
    Train TinyBERT model on edge device with memory-efficient processing.
    Handles both Medical Transcriptions and MIMIC datasets.
    """
    print("Starting TinyBERT training on edge device...")
    
    # Load model and tokenizer with memory-efficient settings
    model, tokenizer = load_bert_model()
    
    # Create a clean mapping for medical specialties
    specialty_mapping = {
        'Cardiovascular / Pulmonary': 'Cardiovascular',
        'SOAP / Chart / Progress Notes': 'Office Notes',
        'Consult - History and Phy.': 'Consult',
        'Physical Medicine - Rehab': 'Physical Medicine',
        'Discharge Summary': 'Discharge',
        'Emergency Room Reports': 'Emergency',
        'Obstetrics / Gynecology': 'Obstetrics',
        'Pediatrics - Neonatal': 'Pediatrics',
        'Hematology - Oncology': 'Hematology',
        'Psychiatry / Psychology': 'Psychiatry',
        'ENT - Otolaryngology': 'ENT',
        'Hospice - Palliative Care': 'Hospice',
        'IME-QME-Work Comp etc.': 'IME-QME'
    }
    
    # Determine dataset type and load appropriate data
    is_mimic = data_path is None or "mimic" in str(data_path).lower()
    
    if is_mimic:
        from dataset.mimic_processor import process_mimic_data
        # Store the data generators for reuse across epochs
        train_gen, val_gen = process_mimic_data(
            batch_size=8,
            max_samples=max_samples
        )
        # Convert generator output to lists for multiple epoch access
        train_data = []
        print("Processing MIMIC training data...")
        for texts, sensitive_features in train_gen:
            train_data.append((texts, sensitive_features))
            if len(train_data) * 8 >= max_samples:  # Using batch_size=8
                break
        print(f"Processed {len(train_data)} batches of MIMIC data")
    else:
        from dataset.mt_processor import process_medical_transcriptions_data
        X_train, _, y_train, _, sf_train, _ = process_medical_transcriptions_data(
            data_path,
            batch_size=8
        )
        
        # Limit dataset size for edge device
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        sf_train = sf_train[:max_samples]
        
        # Clean labels by stripping whitespace and standardizing format
        y_train = [label.strip() for label in y_train]
        clean_specialties = [specialty.strip() for specialty in medical_specialties]
        
        # Map specialties using the specialty_mapping
        y_train = [specialty_mapping.get(label, label) for label in y_train]
        
        # Create label encoder for medical specialties
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        # Fit encoder on cleaned specialties to ensure consistent mapping
        label_encoder.fit(clean_specialties)
        
        try:
            # Transform the labels, handling any unknown labels
            y_train_encoded = []
            for label in y_train:
                try:
                    idx = clean_specialties.index(label)
                    y_train_encoded.append(idx)
                except ValueError:
                    print(f"Warning: Unknown specialty '{label}', mapping to 'General Medicine'")
                    idx = clean_specialties.index('General Medicine')
                    y_train_encoded.append(idx)
            y_train_encoded = np.array(y_train_encoded)
            
        except Exception as e:
            print(f"Error encoding labels: {e}")
            print("Available specialties:", clean_specialties)
            print("Sample of actual labels:", y_train[:10])
            raise
    
    # Configure optimizer and loss function
    optimizer = tf.optimizers.Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        name='adam_opt'
    )
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # Create model save directory
    model_save_dir = os.path.join(os.getcwd(), "tinybert_model")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Training metrics
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        if is_mimic:
            # Training on MIMIC data using stored batches
            for texts, sensitive_features in train_data:
                # Tokenize batch
                inputs = tokenizer(
                    texts.tolist(),
                    padding=True,
                    truncation=True,
                    return_tensors="tf",
                    max_length=64
                )
                
                # Convert gender labels to numeric
                labels = tf.convert_to_tensor(
                    sensitive_features['gender'].map({'male': 0, 'female': 1}).values,
                    dtype=tf.int32
                )
                
                # Train step
                loss, logits = train_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=labels,
                    loss_fn=loss_fn
                )
                
                # Update metrics
                train_acc_metric.update_state(labels, logits)
                
                total_loss += loss
                steps += 1
                
                if steps % 10 == 0:
                    accuracy = train_acc_metric.result().numpy()
                    print(f"Step {steps}, Loss: {total_loss/steps}, Accuracy: {accuracy}")
                
                # Force garbage collection
                gc.collect()
        else:
            # Training on Medical Transcriptions data
            # Process in smaller batches
            batch_size = 8
            for i in range(0, len(X_train), batch_size):
                batch_texts = X_train[i:i + batch_size]
                batch_labels = y_train_encoded[i:i + batch_size]  # Use encoded labels
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts.tolist(),
                    padding=True,
                    truncation=True,
                    return_tensors="tf",
                    max_length=64
                )
                
                # Convert labels to tensor (already numeric from label encoder)
                labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)
                
                # Train step
                loss, logits = train_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=labels,
                    loss_fn=loss_fn
                )
                
                # Update metrics
                train_acc_metric.update_state(labels, logits)
                
                total_loss += loss
                steps += 1
                
                if steps % 10 == 0:
                    accuracy = train_acc_metric.result().numpy()
                    print(f"Step {steps}, Loss: {total_loss/steps}, Accuracy: {accuracy}")
                
                # Force garbage collection
                gc.collect()
        
        # Calculate epoch metrics
        epoch_loss = total_loss / steps
        epoch_accuracy = train_acc_metric.result().numpy()
        
        # Update best metrics
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}")
        
        # Reset metrics
        train_acc_metric.reset_state()
        gc.collect()
    
    # Save the model
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    
    # Save label encoder mapping if using MT dataset
    if not is_mimic:
        import json
        label_mapping = {label: idx for idx, label in enumerate(clean_specialties)}
        with open(os.path.join(model_save_dir, "label_mapping.json"), "w") as f:
            json.dump(label_mapping, f)
        # Also save specialty mapping for reference
        with open(os.path.join(model_save_dir, "specialty_mapping.json"), "w") as f:
            json.dump(specialty_mapping, f)
    
    return {
        'loss': float(epoch_loss),
        'accuracy': float(epoch_accuracy),
        'best_accuracy': float(best_accuracy),
        'best_loss': float(best_loss)
    }

if __name__ == "__main__":  
    # Train MobileNet
    chest_xray_path = "dataset/chest_xray/train"
    history_mobilenet = train_mobilenet_edge(
        chest_xray_path,
        epochs=2,
        samples_per_class=50  # 50 samples per class for quick training
    )
    
    # Clear memory before next training
    gc.collect()
    
    # Train T5
    mt_path = "dataset/mt"
    loss_t5 = train_t5_edge(
        mt_path,
        epochs=1,
        max_samples=100  # Limited samples for T5
    )
    
    # Clear memory before next training
    gc.collect()
    
    # Train TinyBERT
    loss_bert = train_bert_edge(
        mt_path,
        epochs=1,
        max_samples=150  # Limited samples for BERT
    )
