# MCYT Online Signature Verification

**Pattern Recognition Exercise 4**

## 👥 Team & Roles

| Role | Members | Task |
| :--- | :--- | :--- |
| **E (Lead)** | **Lishang Xu** | **Project architecture, Integration, Main pipeline & Final Report** |
| **A (Data)** | [Name A] | Data loader implementation (TSV parsing) & Ground Truth handling |
| **B (Feat)** | [Name B] | Dynamic feature extraction (Velocity/Pressure) & Z-Score Normalization |
| **C (Algo)** | [Name C] | Dynamic Time Warping (DTW) algorithm implementation |
| **D (Eval)** | [Name D] | Performance evaluation metrics (EER) & DET Curve plotting |

## 📁 Project Structure

```text
signature-verification-2025/
├── src/
│   ├── dtw_algo.py       # DTW distance calculation (Role C)
│   ├── features.py       # Feature extraction & normalization (Role B)
│   └── loader.py         # Data loading utilities (Role A)
├── data/                 # Local dataset (ignored by git)
├── results/              # Generated plots and reports
├── main.py               # Main execution script (Role E)
├── det_curve.png         # Generated Detection Error Trade-off curve
├── requirements.txt      # Python dependencies
└── README.md
```

## ⚙️ Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
# .\.venv\Scripts\activate    # (Windows)
# source .venv/bin/activate   # (Linux/macOS)

# Install required packages
pip install -r requirements.txt
```

## 💾 Data

Please download the **MCYT Dataset** from ILIAS and place it under the `data/` directory. [cite_start]The structure must be as follows [cite: 337-342]:

```text
data/
  ├─ enrollment/          # 5 genuine signatures per user (e.g., 001-g-01.tsv)
  ├─ verification/        # 45 signatures per user (20 genuine, 25 forgeries)
  ├─ gt.tsv               # Ground Truth labels
  └─ writers.tsv          # List of writers
```

*Note: The `data/` folder is excluded from version control via `.gitignore`.*

## 🧰 Pipeline & Features

The system implements a functional verification pipeline:

1.  [cite_start]**Data Loading**: Parses TSV files containing time-series data `[t, x, y, pressure, ...]` [cite: 352-360].
2.  **Feature Extraction**:
      * **Dynamic Features**: Ignores absolute `(x, y)` coordinates to resist skilled forgeries.
      * **Velocity**: Computes horizontal ($v_x$) and vertical ($v_y$) velocity derivatives.
      * **Smoothing**: Applies rolling window smoothing to reduce jitter.
3.  **Normalization**: Applies **Z-Score Normalization** to align feature scales.
4.  **Matching**: Uses **Dynamic Time Warping (DTW)** with length normalization to compute dissimilarity scores.

## 🚀 Running the Project

To run the full verification process on the 30-user subset:

```bash
# Run the main script from the root directory
python main.py
```

**Expected Output:**
The script will process users 001-030, calculate distances, and output the final Equal Error Rate (EER).

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

## 📊 Report & Results

  * **Final EER**: 8.08%
  * **Visualization**: A DET (Detection Error Trade-off) curve is automatically generated as `det_curve.png`.
  * **Discussion**: The low EER demonstrates that dynamic features (velocity/pressure) are significantly more discriminative than static shape information for skilled forgeries.

-----

© 2025 · University of Fribourg · Pattern Recognition