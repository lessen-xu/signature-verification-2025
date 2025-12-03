# signature-verification-2025
Online Signature Verification on MCYT dataset - Pattern Recognition Exercise 4

PART A – Data Loader

The data loader provides a unified interface to access all signatures in the MCYT dataset.
What was implemented：
1.Loading of all writers listed in writers.tsv

2.Parsing of signature files (.tsv) from both enrollment/ and verification/

3.Integration of ground-truth labels from gt.tsv

4.ID normalization (e.g., 1 → 001)

5.Returning a standardized structure that other modules can directly use