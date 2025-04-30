import os
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import logging

logger = logging.getLogger(__name__)

# Add at the top of the file with other imports
medical_specialties = [
    'Surgery', 'Consult', 'Cardiovascular', 'Orthopedic', 'Radiology',
    'General Medicine', 'Gastroenterology', 'Neurology', 'Urology', 'ENT',
    'Hematology', 'Obstetrics', 'Neurosurgery', 'Pediatrics', 'Oncology',
    'Dental', 'Psychiatry', 'Ophthalmology', 'Nephrology', 'Gynecology',
    'Respiratory', 'Emergency', 'Podiatry', 'Dermatology', 'Pain Management',
    'Allergy', 'Bariatrics', 'Endocrinology', 'IME-QME', 'Chiropractic',
    'Physical Medicine', 'Sleep Medicine', 'Psychiatry-Psychology', 'LAB',
    'Rheumatology', 'Letters', 'Discharge', 'Cosmetic', 'Hospice',
    'Speech', 'Autopsy', 'Diets', 'Office Notes', 'SOAP', 'Rehab'
]

def load_mobilenet_model():
    model_path = os.path.join(os.getcwd(), 'mobilenet_model.keras')
    if not os.path.exists(model_path):
        # Build and compile the model with 3-channel input
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.save(model_path, save_format='keras')
        print(f"[Loader] New MobileNet model created and saved to {model_path}")
    else:
        # Load the model without compiling to reset the optimizer
        model = tf.keras.models.load_model(model_path, compile=False)
        # Recompile the model with a fresh optimizer
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(f"[Loader] MobileNet model loaded and recompiled from {model_path}")
    return model

def load_t5_model():
    """
    Memory-efficient T5 model loading for IoT devices.
    Uses t5-small with minimal memory footprint and batch processing.
    """
    model_dir = os.path.join(os.getcwd(), 't5_small')
    os.makedirs(model_dir, exist_ok=True)
    
    # Configure TensorFlow for memory efficiency
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            # Limit GPU memory
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]  # Reduced from 1024
            )
    except:
        logger.info("No GPU available, using CPU only")
        
    # Configure TensorFlow CPU memory
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.set_soft_device_placement(True)
    except:
        logger.warning("Could not set all TensorFlow memory configurations")
    
    try:
        if os.path.exists(os.path.join(model_dir, "config.json")):
            # Load locally saved model with minimal memory settings
            model = TFT5ForConditionalGeneration.from_pretrained(
                model_dir,
                from_pt=False,
                use_cache=False
            )
            tokenizer = T5Tokenizer.from_pretrained(
                model_dir,
                model_max_length=64  # Further reduced from 128 to save memory
            )
            
            # Set generation config after model initialization
            model.generation_config.max_new_tokens = 64
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.num_beams = 1  # Disable beam search to save memory
            
            return model, tokenizer
            
        # Download and save if not found locally
        model = TFT5ForConditionalGeneration.from_pretrained(
            't5-small',  # Using smallest T5 variant
            from_pt=False,
            use_cache=False
        )
        tokenizer = T5Tokenizer.from_pretrained(
            't5-small',
            model_max_length=64
        )
        
        # Set generation config
        model.generation_config.max_new_tokens = 64
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.num_beams = 1
        
        # Save model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading T5 model: {str(e)}")
        raise

def load_bert_model():
    # Use consistent path within the project directory
    model_dir = os.path.join(os.getcwd(), "tinybert_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Configure TensorFlow for memory efficiency
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
            )
    except:
        logger.info("No GPU available, using CPU only")
    
    try:
        from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
        
        if os.path.exists(os.path.join(model_dir, "config.json")):
            # Load locally saved model
            model = TFAutoModelForSequenceClassification.from_pretrained(
                model_dir,
                from_pt=False,  # Local model will be in TF format
                use_cache=False
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                model_max_length=64
            )
            return model, tokenizer
            
        # Download and save if not found locally
        model = TFAutoModelForSequenceClassification.from_pretrained(
            'huawei-noah/TinyBERT_General_4L_312D',
            from_pt=True,  # Convert from PyTorch weights
            num_labels=len(medical_specialties),  # Set number of output classes
            use_cache=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'huawei-noah/TinyBERT_General_4L_312D',
            model_max_length=64
        )
        
        # Configure model for training
        model.config.update({
            "learning_rate": 1e-4,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16
        })
        
        # Compile the model with string identifier for optimizer
        model.compile(
            optimizer="adam",  # Using string identifier instead of instance
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        
        # Save model and tokenizer in TensorFlow format
        model.save_pretrained(model_dir, saved_model=True)
        tokenizer.save_pretrained(model_dir)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading TinyBERT model: {str(e)}")
        raise