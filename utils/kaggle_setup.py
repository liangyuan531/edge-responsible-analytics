import os
import shutil

# Get the absolute path to the kaggle.json file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_CONFIG_DIR = os.path.join(ROOT_DIR, '..', 'kaggle_credentials')

# Setup Kaggle credentials before importing the kaggle module
def setup_kaggle_credentials():
    """
    Setup Kaggle credentials before any Kaggle API operations
    """
    try:
        # Create Kaggle config directory if it doesn't exist
        kaggle_dir = os.path.expanduser('~/.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Copy kaggle.json to the default location
        kaggle_json_src = os.path.join(KAGGLE_CONFIG_DIR, 'kaggle.json')
        kaggle_json_dst = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_json_dst):
            shutil.copy2(kaggle_json_src, kaggle_json_dst)
            # Set appropriate permissions
            os.chmod(kaggle_json_dst, 0o600)
            
    except Exception as e:
        raise Exception(f"Failed to setup Kaggle credentials: {str(e)}")