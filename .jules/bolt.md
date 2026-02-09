## 2025-02-12 - [Fake Incremental Computation]
**Learning:** The `IncrementalMelSpectrogram` claimed to reuse cached frames but was actually recomputing the entire audio window (O(N)) and then overwriting the cached part.
**Action:** When working on "incremental" or "caching" features, always benchmark the processing time vs data size. True incremental processing should scale with `new_data`, not `total_data`.
