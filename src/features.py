import numpy as np
import pandas as pd

def normalize_signature(df):
    """
    Normalizes the signature (e.g., centering x and y coordinates).
    """
    # TODO: Member B implementation
    # Example: df['x'] = df['x'] - df['x'].mean()
    return df

def extract_global_features(df):
    """
    Extracts a fixed-length vector of global features for ML models.
    
    Args:
        df (pd.DataFrame): Signature data.
        
    Returns:
        np.array: Array of features (e.g., [total_duration, avg_pressure]).
    """
    # TODO: Member B implementation
    # Example features: Total duration, Average Pressure, Total path length
    features = []
    return np.array(features)

def extract_local_features(df):
    """
    Extracts time-series features (adding derivatives like velocity).
    
    Returns:
        np.array: Sequence of feature vectors (T x D).
    """
    # TODO: Member B implementation
    # Calculate Velocity (vx, vy) and Acceleration (ax, ay)
    return df.values