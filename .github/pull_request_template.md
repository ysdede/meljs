## Summary
- What changed:
- Why this change is needed:

## Scope Guard
- [ ] This PR is scoped to one concern (no unrelated cleanup/churn).
- [ ] I did not change fragile logic without targeted tests.

## Fragile Areas Touched
- [ ] Mel spectrogram pipeline (`src/mel.js`)
- [ ] Incremental mel caching / overlap (`src/mel.js` IncrementalMelSpectrogram)
- [ ] FFT / STFT implementation (`src/mel.js`)
- [ ] Parakeet variant processors (`src/parakeet-variants.js`)
- [ ] Benchmark scripts (`tests/benchmark_variants.mjs`)
- [ ] None of the above

## Verification
- [ ] `npm test`
- [ ] Added/updated targeted tests for touched fragile areas
- [ ] Verified no regression in mel output accuracy (ONNX reference cross-validation)
- [ ] Verified no regression in incremental caching behavior

### Test Evidence
Paste command output snippets here (or link CI runs):

## Risk and Rollback
- Risk level: `low` / `medium` / `high`
- Rollback plan (single commit/PR to revert if needed):

## Related Issues
- Closes/Fixes:
- Follow-ups (if any):
