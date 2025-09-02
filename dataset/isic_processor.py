import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def load_isic_data(max_samples=1000):  # Limit samples for IoT devices
    """
    Load a subset of the ISIC 2019 dataset from Hugging Face.
    """
    try:
        # Load dataset with streaming enabled to save memory
        ds = load_dataset(
            "Anwarkh1/ISIC_2019_Training_Input",
            streaming=True
        )
        
        # Take only the first max_samples
        ds_subset = ds['train'].take(max_samples)
        
        # Convert to lists for DataFrame creation
        data = {
            'image': [],
            'label': [],  # dx field
            'age': [],    # age_approx field
            'gender': [], # sex field
            'site': [],   # anatom_site_general field
            'lesion_id': []
        }
        
        # Process samples one by one
        for sample in ds_subset:
            data['image'].append(sample['image'])
            data['label'].append(sample['dx'])
            data['age'].append(sample['age_approx'])
            data['gender'].append(sample['sex'])
            data['site'].append(sample['anatom_site_general'])
            data['lesion_id'].append(sample['lesion_id'])
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Process age into groups for better privacy protection
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 18, 30, 50, 70, 100],
                                labels=['0-18', '19-30', '31-50', '51-70', '70+'])
        
        logger.info(f"Loaded {len(df)} samples from ISIC dataset")
        return df
        
    except Exception as e:
        logger.error(f"Error loading ISIC data: {e}")
        raise

def preprocess_image(image):
    """
    Preprocess the image data from the dataset.
    """
    try:
        # Convert PIL image to array
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        
        # Convert to float32 and ensure 3 channels
        img_array = tf.cast(img_array, tf.float32)
        if img_array.shape[-1] == 1:
            img_array = tf.image.grayscale_to_rgb(img_array)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
            
        # Resize to target size
        img_array = tf.image.resize(img_array, (224, 224))
        
        # Preprocess for MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array.numpy()  # Convert to numpy array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def process_data_in_batches(df_subset, batch_size=16):
    """
    Process data in batches to manage memory efficiently.
    """
    total_samples = len(df_subset)
    for start_idx in range(0, total_samples, batch_size):
        try:
            end_idx = min(start_idx + batch_size, total_samples)
            batch_df = df_subset.iloc[start_idx:end_idx]
            
            images = []
            labels = []
            sensitive_features = []
            
            for _, row in batch_df.iterrows():
                try:
                    # Process image
                    img_array = preprocess_image(row['image'])
                    if img_array is not None and img_array.shape == (224, 224, 3):
                        images.append(img_array)
                        
                        # Process label (assuming multi-class classification)
                        labels.append(row['label'])
                        
                        # Create sensitive features dictionary
                        sensitive = {
                            'gender': row['gender'],
                            'age_group': row['age_group']
                        }
                        sensitive_features.append(sensitive)
                except Exception as e:
                    logger.error(f"Error processing individual image: {e}")
                    continue
            
            if images:  # Only yield if we have valid images
                yield (np.array(images), 
                       np.array(labels), 
                       pd.DataFrame(sensitive_features))
                   
        except Exception as e:
            logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
            continue

def prepare_dataset(df, test_size=0.2, batch_size=16):
    """
    Prepare the dataset for training and testing with batch processing.
    """
    try:
        # Split the data
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
        
        return (process_data_in_batches(train_df, batch_size), 
                process_data_in_batches(val_df, batch_size))
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise

def process_isic_data(batch_size=16, max_samples=1000):
    """
    Process ISIC data with batch processing.
    Returns generators for training and validation data.
    
    Args:
        batch_size: Size of batches for processing
        max_samples: Maximum number of samples to load (for IoT devices)
    """
    try:
        # Load the dataset with sample limit
        df = load_isic_data(max_samples=max_samples)
        
        # Prepare dataset with batch processing
        train_generator, val_generator = prepare_dataset(df, batch_size=batch_size)
        
        return train_generator, val_generator
        
    except Exception as e:
        logger.error(f"Error processing ISIC data: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the data loading and processing with limited samples
    train_gen, val_gen = process_isic_data(batch_size=16, max_samples=1000)
    
    # Test first batch
    X_batch, y_batch, sensitive_batch = next(train_gen)
    
    print(f"Batch shapes:")
    print(f"Images: {X_batch.shape}")
    print(f"Labels: {y_batch.shape}")
    print(f"Sensitive features: {sensitive_batch.shape}")
