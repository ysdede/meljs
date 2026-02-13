## 2024-05-23 - Mel Filterbank Sparsity
**Learning:** Mel filterbanks are extremely sparse (~98% zeros). Iterating over the full frequency range for each mel bin is a major bottleneck.
**Action:** Precompute start/end indices for non-zero values in filterbanks to skip zero multiplications. This yielded a ~3x speedup.
