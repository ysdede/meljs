# .jules/bolt.md

## 2024-05-21 - Sparse Mel Filterbank
Learning: Mel filterbank matrix is ~98% sparse; iterating over full matrix in hot loop is a major bottleneck (3.75x slow).
Action: Always check for sparsity in numerical processing loops; precompute indices if possible.
