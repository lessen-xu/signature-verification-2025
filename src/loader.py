import os
import pandas as pd
import numpy as np

def load_signature(file_path):
    """
    Loads a single signature TSV file.
    
    Args:
        file_path (str): Path to the .tsv file.
        
    Returns:
        pd.DataFrame: DataFrame containing signature data (t, x, y, pressure, etc.).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Reading TSV file (Tab Separated Values)
    # The dataset contains columns: t, x, y, pressure, penup, azimuth, inclination
    df = pd.read_csv(file_path, sep='\t')
    return df

def load_ground_truth(gt_path):
    """
    Loads the ground truth file (gt.tsv).
    
    Returns:
        pd.DataFrame: DataFrame with columns [filename, label] (or similar).
    """
    return pd.read_csv(gt_path, sep='\t', header=None, names=['filename', 'label'])

def get_user_signatures(user_id, base_path='data'):
    """
    Retrieves file paths for a specific user.
    
    Args:
        user_id (str): The user ID (e.g., '001').
        base_path (str): Root data directory.
        
    Returns:
        dict: {'enrollment': [paths], 'verification': [paths]}
    """
    # Helper logic to verify file structure based on MCYT dataset
    enrollment_dir = os.path.join(base_path, 'enrollment')
    verification_dir = os.path.join(base_path, 'verification')
    
    # TODO: Member A implementation to list specific files for the user
    # e.g., 001-g-01.tsv to 001-g-05.tsv for enrollment
    pass