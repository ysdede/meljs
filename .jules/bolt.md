## 2026-02-20 - Mel Filterbank Sparsity
Learning: The Mel filterbank is ~98.5% sparse, and iterating over the full dense matrix for each mel bin is a major bottleneck (78% of runtime).
Action: Use sparse matrix representations (e.g., precomputed start/end indices) for filterbank operations in audio processing pipelines.
