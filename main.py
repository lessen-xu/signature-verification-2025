import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- IMPORTS ---
try:
    from src.loader import load_signature_tsv as load_signature
except ImportError:
    try:
        from src.loader import load_signature
    except ImportError:
        print("[CRITICAL] Could not import load_signature from src.loader")
        exit(1)

try:
    from src.dtw import calculate_dtw_distance
except ImportError:
    try:
        from src.dtw import calculate_dtw_distance
    except ImportError:
        print("[CRITICAL] Could not import calculate_dtw_distance from src")
        exit(1)

# --- HELPER FUNCTIONS ---

def compute_eer(genuine_scores, impostor_scores):
    if not genuine_scores or not impostor_scores:
        return 0.0, [], []
    
    start = min(np.min(genuine_scores), np.min(impostor_scores))
    end = max(np.max(genuine_scores), np.max(impostor_scores))
    thresholds = np.linspace(start, end, 2000)
    
    min_diff = float('inf')
    eer = 0.0
    
    frr_list = []
    far_list = []
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    for threshold in thresholds:
        frr = np.mean(genuine_scores > threshold)
        far = np.mean(impostor_scores <= threshold)
        
        frr_list.append(frr)
        far_list.append(far)
        
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
            
    return eer * 100, frr_list, far_list

def plot_det_curve(frr_list, far_list, filename="det_curve.png"):
    plt.figure(figsize=(8, 8))
    plt.plot(far_list, frr_list, label='DTW System', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Curve')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def zscore_normalize(seq_array):
    mean = np.mean(seq_array, axis=0)
    std = np.std(seq_array, axis=0)
    std[std == 0] = 1.0 
    return (seq_array - mean) / std

def extract_features(df):
    if len(df.columns) >= 4:
        data = df.iloc[:, [1, 2, 3]].astype(float) 
    else:
        data = pd.DataFrame(df.values[:, :3])

    data_smooth = data.rolling(window=3, min_periods=1, center=True).mean()
    matrix = data_smooth.values
    
    vx = np.gradient(matrix[:, 0])
    vy = np.gradient(matrix[:, 1])
    p = matrix[:, 2]
    
    return np.column_stack([vx, vy, p])

def load_ground_truth(data_path):
    gt_path = os.path.join(data_path, 'gt.tsv')
    if not os.path.exists(gt_path):
        return None
    
    gt_map = {}
    try:
        df = pd.read_csv(gt_path, sep='\t', header=None, dtype=str)
        for _, row in df.iterrows():
            fname = row[0].replace('.tsv', '').strip()
            label = row[1].strip().lower()
            gt_map[fname] = label
        return gt_map
    except:
        return None

# --- MAIN EXECUTION ---

def main():
    print("=== MCYT Signature Verification (30 Writers) ===")
    
    data_path = 'data'
    if not os.path.exists(data_path):
        print("[Error] 'data' folder not found.")
        return

    gt_map = load_ground_truth(data_path)
    
    # Process only first 30 users as per exercise requirements
    users = [f"{i:03d}" for i in range(1, 31)]
    
    all_genuine_distances = []
    all_forgery_distances = []

    print(f"Processing {len(users)} users...")

    for user_id in users:
        enrollment_dir = os.path.join(data_path, 'enrollment')
        verification_dir = os.path.join(data_path, 'verification')
        
        # Load Enrollment
        refs = []
        for i in range(1, 6): 
            path = os.path.join(enrollment_dir, f"{user_id}-g-{i:02d}.tsv")
            if os.path.exists(path):
                df = load_signature(path)
                seq = extract_features(df)
                norm_seq = zscore_normalize(seq)
                refs.append(norm_seq)
        
        if not refs: 
            continue

        # Load Verification
        for i in range(1, 46):
            filename_base = f"{user_id}-{i:02d}"
            path = os.path.join(verification_dir, f"{filename_base}.tsv")
            if not os.path.exists(path): 
                continue
            
            is_genuine = False
            if gt_map:
                label = gt_map.get(filename_base)
                if label and (label.startswith('g') or label == '1'):
                    is_genuine = True
                elif label and (label.startswith('f') or label == '0'):
                    is_genuine = False
                else:
                    is_genuine = (i <= 20)
            else:
                is_genuine = (i <= 20)

            test_df = load_signature(path)
            test_seq = extract_features(test_df)
            norm_test = zscore_normalize(test_seq)
            
            dists = []
            for ref_seq in refs:
                d = calculate_dtw_distance(ref_seq, norm_test)
                d = d / (len(ref_seq) + len(norm_test))
                dists.append(d)
            
            score = min(dists)

            if is_genuine:
                all_genuine_distances.append(score)
            else:
                all_forgery_distances.append(score)
        
        print(f"User {user_id} done.")

    print("\n=== Final Results ===")
    print(f"Genuine Samples: {len(all_genuine_distances)}")
    print(f"Forgery Samples: {len(all_forgery_distances)}")
    
    if all_genuine_distances:
        eer, frr, far = compute_eer(all_genuine_distances, all_forgery_distances)
        print("------------------------------------------------")
        print(f"FINAL EER: {eer:.2f}%")
        print("------------------------------------------------")
        plot_det_curve(frr, far)
        print("[Info] det_curve.png saved.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()