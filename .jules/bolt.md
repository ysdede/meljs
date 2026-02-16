## 2024-05-22 - Sparse Mel Filterbank Optimization
**Learning:** The Mel filterbank matrix is ~98.5% sparse. Using a dense matrix multiplication iterates over thousands of zeros unnecessarily.
**Action:** Precompute start/end indices for non-zero elements in the constructor and use them to constrain the inner loop. This yields a ~2.3x speedup for the entire pipeline.
