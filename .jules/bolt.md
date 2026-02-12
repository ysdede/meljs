## 2024-05-22 - Mel Filterbank Sparsity Optimization
**Learning:** The Mel filterbank matrix is triangular and highly sparse (mostly zeros). Standard matrix multiplication iterates over all frequency bins (e.g., 257) for each mel bin, resulting in many zero multiplications.
**Action:** By precomputing the start and end indices of non-zero values for each mel filter, we can restrict the inner loop to only the valid range. This reduced processing time from ~834ms to ~232ms for 60s of audio (~3.6x speedup). Always look for sparsity in fixed transform matrices.
