# Exercise 4: Online Signature Verification (MCYT)

Date: December 6, 2025
Group: 4

## 1. Team Members & Roles

| Role           | Member      | Responsibility                                                    |
| :------------- | :---------- | :---------------------------------------------------------------- |
| Lead (E)       | Lishang Xu  | System design, integration, pipeline implementation, final report |
| Data (A)       | Bole Yi     | Dataset loading (TSV parsing) and ground truth mapping            |
| Features (B)   | Songzhi Liu | Dynamic feature extraction (velocity/pressure) and normalization  |
| Algorithm (C)  | Yuting Zhu  | DTW implementation                                                |
| Evaluation (D) | Jules       | EER metric and DET curve generation                               |

---

## 2. Introduction

The goal of this exercise was to build an online signature verification system using the MCYT Baseline Corpus. The task focuses on distinguishing genuine signatures from skilled forgeries — forgeries created by people who had the opportunity to practice the target signature.

Unlike offline methods based on static images, online verification uses time-series data such as pen position, pressure, and velocity. The performance is evaluated using the Equal Error Rate (EER), and the system was tested on 30 users from the dataset.

---

## 3. Methodology

Our system follows a staged pipeline from data loading to scoring.

### 3.1 Data Loading & Ground Truth

We implemented a TSV parser to read the time-series format, which includes:

```
[t, x, y, pressure, azimuth, inclination]
```

Because file order does not always map cleanly to genuine/forged labels, we used `gt.tsv` as the reference for correct labeling.

---

### 3.2 Feature Extraction

Initial testing using raw coordinates `(x, y)` resulted in poor performance (EER ~50%). Skilled forgers can reproduce the visible shape of a signature, so static spatial features weren't sufficient.

To improve discrimination, we switched to **dynamic features** that capture writing behavior rather than appearance:

* Velocity:

  * (v_x = \frac{dx}{dt})
  * (v_y = \frac{dy}{dt})
* Pressure
* A rolling smoothing filter (window size = 3) to reduce noise

Final feature vector:

```
[vx, vy, pressure]
```

---

### 3.3 Normalization

We applied Z-score normalization to ensure that different feature scales (e.g., velocity vs. pressure) contribute fairly to the comparison.

---

### 3.4 Matching Algorithm: DTW

We used Dynamic Time Warping (DTW) to measure sequence similarity.

* Reference signatures: 5 genuine samples per user
* Score: minimum DTW distance among the five
* Distance was normalized by sequence length to avoid bias toward shorter signatures

---

## 4. Experiments & Results

Evaluation was performed on the first 30 users:

* 600 genuine samples (20/user)
* 750 forgeries (25/user)

### 4.1 Quantitative Result

| Metric                 | Value |
| :--------------------- | :---- |
| Equal Error Rate (EER) | 8.08% |

### 4.2 Visualization

The DET curve illustrates the trade-off between false rejection and false acceptance. The curve is close to the origin, indicating good system performance.

*(Curve omitted here, see report file: `det_curve.png`)*

---

## 5. Discussion

Our results highlight several key observations:

1. **Dynamic features matter:**
   Using only shape-based spatial features failed to separate genuine signatures from practiced forgeries. Incorporating velocity and pressure allowed the system to capture writing rhythm and hand motion, reducing EER from ~50% to 8.08%.

2. **Normalization is essential:**
   Without Z-score standardization, pressure values dominated the distance metric, degrading performance.

3. **Ground truth alignment:**
   Properly parsing `gt.tsv` avoided mislabeling errors that would have invalidated evaluation.

---

## 6. Conclusion

We developed an online signature verification system using MCYT and achieved an EER of **8.08%**. The results confirm that dynamic handwriting behavior — not just the visual shape — plays a critical role in authentication accuracy, and DTW combined with well-selected features provides a strong baseline against skilled forgeries.

---
