# signature-verification-2025
Online Signature Verification on MCYT dataset - Pattern Recognition Exercise 4
# Signature Verification — Pattern Recognition Exercise 4

This repository contains our implementation of an **On-line Signature Verification** system based on the MCYT dataset.
The goal is to verify whether a given signature is genuine or a forgery based on time-series data (x, y, pressure, azimuth, inclination).

## 📁 Project Structure

```text
src/
  ├── A_loader/        # Data ingestion & parsing (TSV files)
  ├── B_features/      # Feature extraction (velocity, acceleration, global stats)
  ├── C_baseline_dtw/  # Baseline algorithm: Dynamic Time Warping (DTW)
  ├── D_ml_model/      # Machine Learning approach (SVM / HMM / Random Forest)
  └── E_evaluation/    # EER calculation & ROC/DET curve plotting

data/                  # MCYT dataset (enrollment/verification folders) - Ignored by Git
results/               # Output plots and reports
main.py                # Main execution script