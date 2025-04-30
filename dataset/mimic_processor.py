import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datasets import load_dataset
import logging
import re
import gc

logger = logging.getLogger(__name__)

def load_mimic_data(max_samples=1000):
    """
    Load a subset of the MIMIC-III dataset from Hugging Face.
    """
    try:
        # Load dataset with streaming enabled to save memory
        ds = load_dataset(
            "Medilora/mimic_iii_diagnosis_anonymous",
            streaming=True
        )
        
        # Take only the first max_samples
        ds_subset = ds['train'].take(max_samples)
        
        # Convert to lists for DataFrame creation
        data = {
            'text': []
        }
        
        # Process samples one by one
        for sample in ds_subset:
            data['text'].append(sample['text'])
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"Loaded {len(df)} samples from MIMIC-III dataset")
        return df
        
    except Exception as e:
        logger.error(f"Error loading MIMIC-III data: {e}")
        raise

def extract_demographics(text):
    """
    Extract demographic information from the text.
    Returns gender, age and other sensitive information if found.
    """
    # Initialize default values
    gender = None
    age = None
    
    # Convert to lowercase for easier matching
    text_lower = text.lower()
    
    # Extract gender using regex patterns
    gender_patterns = [
        r'\b(?:is|was) (?:a|an) (\d+)[\s-]*(?:year|y[./o]|yo).*?(male|female)',
        r'\b(male|female)\s+(?:patient|individual)',
        r'(?:gender|sex):\s*(male|female)',
        r'this\s+(male|female)\s+(?:patient|individual)',
        r'the\s+(?:patient\s+)?(?:is|was)\s+(?:a\s+)?(\d+)[\s-]*(?:year|y[./o]|yo)[\s-]*(?:old\s+)?(male|female)'
    ]
    
    for pattern in gender_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Get gender from the last group that matches 'male' or 'female'
            gender_groups = [g for g in match.groups() if g in ['male', 'female']]
            if gender_groups:
                gender = gender_groups[-1]
                break
    
    # Extract age using regex
    age_patterns = [
        r'\b(\d{1,3})[\s-]*(?:year|y[./o]|yo)[\s-]*(?:old)?(?:\s+|$)',
        r'age[:\s]+(\d{1,3})',
        r'(?:is|was)\s+(?:a|an)\s+(\d{1,3})[\s-]*(?:year|y[./o]|yo)',
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                age = int(match.group(1))
                # Filter out unrealistic ages
                if age < 0 or age > 120:
                    age = None
                else:
                    break
            except ValueError:
                continue
    
    return gender, age

def process_data_in_batches(df_subset, batch_size=32):
    """
    Process data in batches to manage memory efficiently.
    """
    total_samples = len(df_subset)
    for start_idx in range(0, total_samples, batch_size):
        try:
            end_idx = min(start_idx + batch_size, total_samples)
            batch_df = df_subset.iloc[start_idx:end_idx]
            
            texts = []
            sensitive_features = []
            
            for _, row in batch_df.iterrows():
                try:
                    # Extract demographic information
                    gender, age = extract_demographics(row['text'])
                    
                    if gender is not None and age is not None:
                        texts.append(row['text'])
                        
                        # Convert age to age group
                        age_group = pd.cut([age], 
                                         bins=[0, 18, 30, 50, 70, 120],
                                         labels=['0-18', '19-30', '31-50', '51-70', '70+'])[0]
                        
                        # Create sensitive features dictionary
                        sensitive = {
                            'gender': gender,
                            'age_group': age_group
                        }
                        sensitive_features.append(sensitive)
                        
                except Exception as e:
                    logger.error(f"Error processing individual text: {e}")
                    continue
            
            if texts:  # Only yield if we have valid samples
                yield (np.array(texts), 
                      pd.DataFrame(sensitive_features))
                   
        except Exception as e:
            logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
            continue

def prepare_dataset(df, test_size=0.2, batch_size=32):
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

def process_mimic_data(batch_size=32, max_samples=1000):
    """
    Process MIMIC-III data with batch processing.
    Returns generators for training and validation data.
    
    Args:
        batch_size: Size of batches for processing
        max_samples: Maximum number of samples to load (for IoT devices)
    """
    try:
        # Load the dataset with sample limit
        df = load_mimic_data(max_samples=max_samples)
        
        # Prepare dataset with batch processing
        train_generator, val_generator = prepare_dataset(df, batch_size=batch_size)
        
        return train_generator, val_generator
        
    except Exception as e:
        logger.error(f"Error processing MIMIC data: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the data loading and processing with limited samples
    train_gen, val_gen = process_mimic_data(batch_size=32, max_samples=1000)
    
    # Test first batch
    try:
        texts, sensitive_batch = next(train_gen)
        
        print(f"Batch shapes:")
        print(f"Number of texts: {len(texts)}")
        print(f"Sensitive features shape: {sensitive_batch.shape}")
        
        # Print sample of extracted demographics
        print("\nSample of extracted demographics:")
        print(sensitive_batch.head())
        
    except StopIteration:
        print("No valid samples found in the first batch")
