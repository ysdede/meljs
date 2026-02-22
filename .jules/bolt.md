## 2024-05-22 - Mel Filterbank Sparsity Optimization
Learning: The Mel filterbank is a highly sparse triangular matrix (mostly zeros). Dense iteration in `computeRawMel` involves many unnecessary multiplications by zero.
Action: Precompute sparse indices (`start` and `end`) for each mel bin to skip zeros, significantly improving performance (approx ~3-4x speedup).
