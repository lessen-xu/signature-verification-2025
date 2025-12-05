import numpy as np
from scipy.spatial.distance import euclidean

def calculate_dtw_distance(seq1, seq2):
    """
    Computes the Dynamic Time Warping (DTW) distance between two sequences.
    
    Args:
        seq1 (np.array): Reference sequence (N x Features).
        seq2 (np.array): Query sequence (M x Features).
        
    Returns:
        float: The DTW distance (alignment cost).
    """

    n, d1 = seq1.shape
    m, d2 = seq2.shape

    if d1 != d2:
        raise ValueError(f"Feature dimensions mismatch: {d1} vs {d2}")

    # Initialize DP matrix with +inf
    DP = np.full((n + 1, m + 1), np.inf)
    DP[0, 0] = 0.0

    # Fill DP using the DTW recurrence
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])  # Euclidean distance
            DP[i, j] = cost + min(
                DP[i - 1, j],     # insertion
                DP[i, j - 1],     # deletion
                DP[i - 1, j - 1]  # match
            )

    return float(DP[n, m])


#  self-test
if __name__ == "__main__":
    a = np.array([[0, 0], [1, 1], [2, 2]])
    b = np.array([[0, 0], [2, 2]])
    print("DTW =", calculate_dtw_distance(a, b))
