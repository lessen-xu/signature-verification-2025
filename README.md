# MCYT Online Signature Verification

**Pattern Recognition Exercise 4**

## 👥 Team & Roles

| Role | Member | Task |
| :--- | :--- | :--- |
| **E (Lead)** | **Lishang Xu** | **Project Architecture, Integration, Main Pipeline & Final Report** |
| **A (Data)** | **Bole Yi** | Data loader implementation (TSV parsing) & Ground Truth handling |
| **B (Feat)** | **Songzhi Liu** | Dynamic feature extraction (Velocity/Pressure) & Z-Score Normalization |
| **C (Algo)** | **Yuting Zhu** | Dynamic Time Warping (DTW) algorithm implementation |
| **D (Eval)** | **Jules** | Performance evaluation metrics (EER) & DET Curve plotting |

## 📁 Project Structure

```text
signature-verification-2025/
├── src/
│   ├── dtw.py            # DTW distance calculation (Yuting Zhu)
│   ├── features.py       # Feature extraction & normalization (Songzhi Liu)
│   ├── loader.py         # Data loading utilities (Bole Yi)
│   └── models.py         # (Optional) Model definitions
├── data/                 # Local dataset (ignored by git)
├── results/              # Directory for output results
├── main.py               # Main execution script (Lishang Xu)
├── det_curve.png         # Generated Detection Error Trade-off curve
├── requirements.txt      # Python dependencies
└── README.md
```

## ⚙️ Environment Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# .\.venv\Scripts\activate    # (Windows)
# source .venv/bin/activate   # (Linux/macOS)

# 2. Install required packages
pip install -r requirements.txt
```

## 💾 Data

The project uses the **MCYT Baseline Corpus**. Please download the dataset from ILIAS and extract it into the `data/` directory.
The structure must be as follows:

```text
data/
  ├─ enrollment/          # 5 genuine signatures per user (e.g., 001-g-01.tsv)
  ├─ verification/        # 45 signatures per user (20 genuine, 25 forgeries)
  ├─ gt.tsv               # Ground Truth labels
  └─ writers.tsv          # List of writers
```

*Note: The `data/` folder is excluded from version control via `.gitignore`.*

## 🧰 Pipeline & Methodology

To distinguish skilled forgeries from genuine signatures, we implemented the following pipeline:

1.  **Data Loading**: The system parses TSV files containing time-series data `[t, x, y, pressure, ...]` and loads ground truth labels.
2.  **Feature Extraction**:
      * **Dynamic Features**: We discard absolute coordinates `(x, y)` to resist skilled forgeries (who often copy shape perfectly).
      * **Velocity**: We compute horizontal ($v_x$) and vertical ($v_y$) velocity derivatives.
      * **Smoothing**: A rolling window is applied to reduce quantization noise and jitter.
3.  **Normalization**: **Z-Score Normalization** is applied to align feature scales (velocity vs. pressure).
4.  **Matching**: We use **Dynamic Time Warping (DTW)** with length normalization to compute the dissimilarity score between the enrollment references and the verification sample.

## 🚀 Running the Project

To run the verification process on the 30-user subset (Exercise 4 Requirement):

```bash
# Run the main script from the root directory
python main.py
```

**Expected Output:**
The script will process users 001-030, calculate DTW distances, and output the final Equal Error Rate (EER).

```text
=== MCYT Signature Verification (30 Writers) ===
Processing 30 users...
...
=== Final Results ===
Genuine Samples: 600
Forgery Samples: 750
------------------------------------------------
FINAL EER: 8.08%
------------------------------------------------
[Info] det_curve.png saved.
```

## 📊 Results

The system achieves an **EER of 8.08%** on the provided subset of 30 writers.

### DET Curve

The Detection Error Trade-off (DET) curve below illustrates the system's performance trade-off between False Acceptance Rate (FAR) and False Rejection Rate (FRR).

-----

© 2025 · University of Fribourg · Pattern Recognition
