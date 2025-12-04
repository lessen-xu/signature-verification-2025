import numpy as np
import pandas as pd

def normalize_signature(df):
    """
    Normalizes the signature to make it invariant to position and scale.
    1. Centering: Subtracts the mean from x and y coordinates.
    2. Scaling: Divides by the standard deviation to handle different signature sizes.
    """
    # Create a copy to avoid modifying the original data
    df = df.copy()

    # 1. Positional Normalization (Centering)
    # Shift the centroid to (0,0)
    df['x'] = df['x'] - df['x'].mean()
    df['y'] = df['y'] - df['y'].mean()

    # 2. Scale Normalization (Scaling)
    # Scale using standard deviation of x and y to make signatures of different sizes comparable
    # Avoid division by zero
    x_std = df['x'].std()
    y_std = df['y'].std()
    
    if x_std > 0:
        df['x'] = df['x'] / x_std
    if y_std > 0:
        df['y'] = df['y'] / y_std

    return df

def extract_global_features(df):
    """
    Extracts a fixed-length vector of global features for ML models (Role D).
    
    Args:
        df (pd.DataFrame): Signature data.
        
    Returns:
        np.array: Array of features.
    """
    # Ensure time t is monotonic to avoid division by zero issues (though usually it is)
    t = df['t'].values
    x = df['x'].values
    y = df['y'].values
    p = df['pressure'].values

    # 1. Total Duration
    duration = t[-1] - t[0] if len(t) > 0 else 0

    # 2. Total Path Length
    # Calculate Euclidean distance between adjacent points
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_path_length = np.sum(dists)

    # 3. Average Pressure
    avg_pressure = np.mean(p)

    # 4. Average Velocity
    # Total path length / duration
    avg_velocity = total_path_length / duration if duration > 0 else 0

    # 5. Number of points
    n_points = len(df)

    features = [duration, total_path_length, avg_pressure, avg_velocity, n_points]
    return np.array(features)

def extract_local_features(df):
    """
    Extracts time-series features for DTW (Role C).
    Adds derivatives like velocity and acceleration.
    
    Returns:
        np.array: Sequence of feature vectors (T x D).
    """
    # Get original sequences
    t = df['t'].values
    x = df['x'].values
    y = df['y'].values
    p = df['pressure'].values
    
    # === Feature Extraction ===
    
    # 1. Compute First Derivative (Velocity)
    # Use np.gradient for central difference, keeping array length same
    # dx/dt, dy/dt
    dt = np.gradient(t)
    # Prevent division by zero (dt=0), though rare
    dt[dt == 0] = 1e-6 
    
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    
    # 2. Compute Second Derivative (Acceleration)
    # dvx/dt, dvy/dt
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    
    # 3. Pressure Derivative (Rate of change of pressure)
    vp = np.gradient(p, t)

    # === Combine Features ===
    # Stack selected features into a matrix.
    # Features commonly used for DTW: x, y, vx, vy, pressure
    # You can decide whether to include ax, ay based on performance
    
    # Shape: (T, 7) -> [x, y, vx, vy, ax, ay, pressure]
    features_matrix = np.column_stack((
        x, 
        y, 
        vx, 
        vy, 
        ax, 
        ay, 
        p
    ))
    
    # Handle possible NaN or Inf values (e.g. due to dt=0)
    features_matrix = np.nan_to_num(features_matrix)
    
    return features_matrix