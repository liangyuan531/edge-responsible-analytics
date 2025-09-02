import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from utils.kaggle_setup import setup_kaggle_credentials

# Setup credentials before importing kaggle
setup_kaggle_credentials()
import kaggle

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("edge_device.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def download_chest_xray_data():
    logger.info(f"Downloading chest xray data")
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path='dataset/',
            unzip=True
        )
        
    except Exception as e:
        raise Exception(f"Failed to download dataset: {str(e)}")

def preprocess_chest_xray(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def process_chest_xray_data(data_path, batch_size=16):
    logger.info(f"[chest_xray_processor] Processing chest xray data from {data_path}")
    if not os.path.exists(data_path):
        download_chest_xray_data()
    
    # Load and preprocess a batch of images
    images = []
    labels = []
    sensitive_features = []
    
    for label in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(data_path, label)
        logger.info(f"chest xray data from {class_dir}")
        for image_file in os.listdir(class_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_dir, image_file)
                images.append(preprocess_chest_xray(image_path))
                labels.append(0 if label == 'NORMAL' else 1)
                # Example sensitive feature: patient gender (assuming gender info is in the filename)
                sensitive_features.append(0 if 'male' in image_file.lower() else 1)
                
                # Process in batches
                if len(images) == batch_size:
                    yield np.stack(images), np.array(labels), np.array(sensitive_features)
                    images, labels, sensitive_features = [], [], []
    
    # Yield any remaining data
    if images:
        yield np.stack(images), np.array(labels), np.array(sensitive_features)

# Example usage
if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(ROOT_DIR, "datasets/chest_xray/train")
    for X_batch, y_batch, sensitive_batch in process_chest_xray_data(data_path):
        print(f"Processed batch with {len(X_batch)} images")
