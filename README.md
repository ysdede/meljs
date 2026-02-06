# meljs

**Pure JavaScript mel spectrogram, FFT, and audio feature extraction for ML/AI.**

Zero dependencies. Runs in browsers (including Web Workers, WebGPU pipelines) and Node.js. NeMo-compatible output validated against ONNX reference models.

> Built for developers working with speech recognition, audio ML, and ONNX Runtime Web who need fast, accurate audio feature extraction without native dependencies.

## Why meljs?

ONNX Runtime Web provides neural network inference but **not audio preprocessing** — you still need to compute mel spectrograms before feeding audio to ASR/audio models. Most existing solutions are either Python-only, require WASM/native bindings, or aren't validated against production models.

meljs fills this gap:

- **Pure JavaScript** — no WASM, no native bindings, no build step
- **NeMo-compatible** — validated against NVIDIA NeMo's ONNX preprocessor
- **Fast** — 5s of audio processed in ~37ms, with incremental caching for streaming
- **Precision** — Float64 STFT matching ONNX's double-precision pipeline, max error < 3.6e-4 vs ONNX
- **Configurable** — works with any mel bin count, FFT size, hop length, sample rate

## Install

```bash
npm install meljs
```

## Quick Start

### Basic Usage

```js
import { MelSpectrogram } from 'meljs';

// Create processor (NeMo-compatible defaults)
const mel = new MelSpectrogram({ nMels: 128 });

// Process audio (Float32Array, mono, 16kHz)
const { features, length } = mel.process(audioFloat32Array);
// features: Float32Array [128 × nFrames], normalized
// length: number of valid frames
```

### Streaming with Incremental Caching

For real-time applications with overlapping windows, `IncrementalMelSpectrogram` caches the prefix frames and only computes new ones:

```js
import { IncrementalMelSpectrogram } from 'meljs';

const mel = new IncrementalMelSpectrogram({ nMels: 128 });

// First chunk: full computation
const r1 = mel.process(audioWindow1, 0);

// Second chunk: 70% overlap → reuses ~70% of frames
const overlapSamples = Math.floor(audioWindow2.length * 0.7);
const r2 = mel.process(audioWindow2, overlapSamples);
console.log(r2.cachedFrames, r2.newFrames);
// e.g., 347 cached, 153 computed — ~2x speedup

// Reset for new recording
mel.reset();
```

### Low-Level Building Blocks

Use individual functions for custom pipelines:

```js
import {
  hzToMel, melToHz,           // Slaney mel scale conversion
  createMelFilterbank,         // Triangular filterbank matrix
  createPaddedHannWindow,      // Symmetric Hann window
  precomputeTwiddles, fft,     // Radix-2 Cooley-Tukey FFT
  MEL_CONSTANTS,               // NeMo-compatible constants
} from 'meljs';

// Create a custom mel filterbank for 22050 Hz, 1024-point FFT
const fb = createMelFilterbank(80, 22050, 1024);

// Run FFT on a frame
const N = 512;
const tw = precomputeTwiddles(N);
const re = new Float64Array(N);
const im = new Float64Array(N);
// ... fill re with windowed audio frame ...
fft(re, im, N, tw);
```

### Web Worker Usage

meljs works seamlessly in Web Workers for background processing:

```js
// mel.worker.js
import { MelSpectrogram } from 'meljs';
const mel = new MelSpectrogram({ nMels: 128 });

self.onmessage = ({ data }) => {
  const { features, length } = mel.process(new Float32Array(data.audio));
  self.postMessage({ features, length }, [features.buffer]);
};
```

## API Reference

### `MelSpectrogram`

Main class for computing log-mel spectrogram features.

| Option | Default | Description |
|--------|---------|-------------|
| `nMels` | `128` | Number of mel bins (80 or 128 typical) |
| `sampleRate` | `16000` | Audio sample rate in Hz |
| `nFft` | `512` | FFT size |
| `winLength` | `400` | STFT window length in samples |
| `hopLength` | `160` | STFT hop length in samples |
| `preemph` | `0.97` | Pre-emphasis coefficient (0 to disable) |
| `logZeroGuard` | `2^-24` | Guard value for log computation |

**Methods:**
- `process(audio)` — Full pipeline: audio → normalized log-mel features
- `computeRawMel(audio)` — Compute un-normalized log-mel (for custom normalization)
- `normalize(rawMel, nFrames, featuresLen)` — Apply Bessel-corrected normalization
- `samplesToFrames(samples)` — Convert sample count to frame count
- `framesToSamples(frames)` — Convert frame count to sample count

### `IncrementalMelSpectrogram`

Extends `MelSpectrogram` with frame-level caching for streaming.

| Option | Default | Description |
|--------|---------|-------------|
| `boundaryFrames` | `3` | Extra frames to recompute at cache boundary |
| *(plus all MelSpectrogram options)* | | |

**Methods:**
- `process(audio, prefixSamples)` — Returns `{features, length, cached, cachedFrames, newFrames}`
- `reset()` / `clear()` — Clear the frame cache

### Standalone Functions

| Function | Description |
|----------|-------------|
| `hzToMel(freq)` | Convert Hz to Slaney mel scale |
| `melToHz(mel)` | Convert Slaney mel to Hz |
| `createMelFilterbank(nMels?, sampleRate?, nFft?)` | Create triangular mel filterbank |
| `createPaddedHannWindow(winLength?, nFft?)` | Create zero-padded Hann window |
| `precomputeTwiddles(N)` | Precompute FFT twiddle factors |
| `fft(re, im, N, tw)` | In-place radix-2 Cooley-Tukey FFT |

### Constants

```js
import { MEL_CONSTANTS } from 'meljs';
// { SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP_LENGTH, PREEMPH, LOG_ZERO_GUARD, N_FREQ_BINS, DEFAULT_N_MELS }
```

## Accuracy

Validated against NVIDIA NeMo's ONNX preprocessor (`nemo128.onnx`):

| Metric | Value |
|--------|-------|
| Mel filterbank max error vs ONNX | **2.645e-7** |
| Full pipeline max error | **3.6e-4** |
| Full pipeline mean error | **1.1e-5** |
| Test signals validated | Sine, chirp, white noise, speech |

## Performance

Benchmarks on a modern desktop (Node.js):

| Duration | Processing Time | Realtime Factor |
|----------|----------------|-----------------|
| 0.5s | ~3ms | ~160x |
| 1s | ~7ms | ~140x |
| 5s | ~37ms | ~135x |
| 10s | ~70ms | ~140x |

**Incremental caching** (70% overlap streaming scenario):

| Mode | Time | Speedup |
|------|------|---------|
| Full recompute | 71.7ms | — |
| Incremental (cached) | 36.6ms | **~2x** |

## Pipeline Details

The mel spectrogram pipeline exactly matches NeMo's `FilterbankFeatures`:

1. **Pre-emphasis** — `x[n] = x[n] - 0.97 * x[n-1]` (Float32)
2. **Zero-pad** — N_FFT/2 = 256 samples each side
3. **STFT** — Cast to Float64, symmetric Hann window (400→512), 512-point FFT
4. **Power spectrum** — `|real|² + |imag|²` → Float32
5. **Mel filterbank** — Slaney-normalized triangular filterbank matmul
6. **Log** — `log(mel + 2^-24)`
7. **Normalize** — Per-feature mean/variance, Bessel-corrected (N-1 denominator)

NeMo's dither, narrowband augmentation, frame splicing, and pad-to-multiple are all disabled at inference and correctly omitted.

## Testing

```bash
npm test          # Run all tests
npm run test:watch  # Watch mode
```

The test suite includes:
- Constants validation
- Mel scale roundtrip tests
- Filterbank shape and triangle continuity
- Hann window symmetry and padding
- FFT correctness (DC, sinusoid, Parseval's theorem, large sizes)
- Full pipeline correctness and determinism
- Incremental caching accuracy
- ONNX reference cross-validation (mel_reference.json)
- Performance benchmarks

## Compatibility

meljs is used in production by:
- [parakeet.js](https://github.com/ysdede/parakeet.js) — Browser-based ASR with WebGPU
- [boncukjs](https://github.com/ysdede/boncukjs) — Real-time transcription app

It replaces the ONNX preprocessor model (`nemo128.onnx`) in both projects, eliminating a 1.5MB model download and ONNX session overhead.

## License

MIT
