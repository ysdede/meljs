## 2024-05-23 - Sparse Mel Filterbank
Learning: The Mel filterbank matrix is ~98.5% sparse (only ~500 non-zero elements out of ~32k), making dense matrix multiplication extremely inefficient.
Action: Always check for sparsity in fixed transform matrices (like Mel or DCT) and implement sparse iteration if sparsity > 90%.
