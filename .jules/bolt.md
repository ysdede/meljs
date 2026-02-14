## 2024-05-22 - Sparse Mel Filterbank Optimization
**Learning:** The Mel filterbank matrix is ~98% sparse. Iterating over all frequency bins for each Mel filter is highly inefficient. Precomputing start/end indices for non-zero values yields a ~3-4x speedup.
**Action:** In signal processing pipelines, always check for sparse constant matrices (like filterbanks) and optimize loops to skip zeros.
