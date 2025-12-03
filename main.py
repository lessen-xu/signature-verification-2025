import os
import sys
from src.loader import load_signature, load_ground_truth
from src.features import extract_global_features, normalize_signature
from src.dtw_algo import calculate_dtw_distance
# from src.models import SignatureVerifier

def main():
    print("=== MCYT Online Signature Verification System ===")
    
    # 1. Configuration
    DATA_PATH = 'data'
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory '{DATA_PATH}' not found.")
        print("Please extract the MCYT dataset into the 'data' folder.")
        return

    # 2. Example Pipeline (Skeleton)
    try:
        # Step A: Load a sample user (e.g., User 001)
        # [cite_start]Note: These paths verify against the dataset structure [cite: 338, 339]
        ref_path = os.path.join(DATA_PATH, 'enrollment', '001-g-01.tsv')
        test_path = os.path.join(DATA_PATH, 'verification', '001-01.tsv')
        
        print(f"Loading reference: {ref_path}")
        ref_df = load_signature(ref_path)
        
        print(f"Loading test: {test_path}")
        test_df = load_signature(test_path)
        
        # Step B: Preprocessing & Features
        print("Extracting features...")
        ref_df = normalize_signature(ref_df)
        test_df = normalize_signature(test_df)
        
        # Step C: Compare using DTW (Member C's work)
        # Convert DataFrames to numpy arrays for DTW
        dist = calculate_dtw_distance(ref_df.values, test_df.values)
        print(f"Calculated DTW Distance: {dist}")
        
        # Step D: Compare using ML Model (Member D's work)
        # global_feats = extract_global_features(ref_df)
        # classifier = SignatureVerifier()
        # classifier.train([global_feats])
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()