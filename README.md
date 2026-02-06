# meljs

**Pure JavaScript mel spectrogram, FFT, and audio feature extraction for ML/AI.**

Zero dependencies. Runs in browsers (including Web Workers, WebGPU pipelines) and Node.js. NeMo-compatible output validated against ONNX reference models.

> Built for developers working with speech recognition, audio ML, and ONNX Runtime Web who need stateful, cacheable audio feature extraction — something ONNX preprocessor models can't provide.

## Why meljs?

ONNX Runtime Web provides neural network inference but **not audio preprocessing**. Most ASR pipelines ship an ONNX preprocessor model (e.g. `nemo128.onnx`) to compute mel spectrograms, but ONNX models are **stateless black boxes** — every call recomputes everything from scratch. This creates two problems:

1. **No caching for streaming.** In real-time ASR, you process overlapping windows (e.g. 5s window, 1s hop). With an ONNX preprocessor, you recompute the mel spectrogram for the full 5s every time — even though 4s of audio is identical to the previous call.

2. **No pipeline parallelism.** The ONNX preprocessor runs on the same thread/session as inference, so encoding can't start until preprocessing finishes.

meljs solves both:

- **Stateful & cacheable** — `IncrementalMelSpectrogram` caches prefix frames across calls. In a 70% overlap streaming scenario, ~70% of frames are reused from cache, computing only new audio.
- **Background pipeline** — Because it's pure JS, meljs can run in a dedicated Web Worker, continuously producing mel frames as audio arrives. When the encoder needs features, they're already computed — **0.0ms preprocessing latency** in the inference path.
- **Pure JavaScript** — no WASM, no native bindings, no model download, no build step
- **NeMo-compatible** — validated against NVIDIA NeMo's ONNX preprocessor (max error < 3.6e-4)
- **Precision** — Float64 STFT matching ONNX's double-precision pipeline
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

### Web Worker — Continuous Mel Producer

The most powerful pattern: run meljs in a dedicated Web Worker that continuously converts audio chunks into mel frames. The inference thread never waits for preprocessing — features are always ready.

```js
// mel.worker.js — runs in background, fed audio chunks continuously
import { MelSpectrogram, MEL_CONSTANTS } from 'meljs';

const mel = new MelSpectrogram({ nMels: 128 });
let audioBuffer = new Float32Array(0);
let melBuffer = null;  // accumulates raw mel frames
let totalFrames = 0;

self.onmessage = ({ data }) => {
  if (data.type === 'push_audio') {
    // Append new audio, compute new mel frames incrementally
    // ~0.5ms per 80ms audio chunk
    const chunk = new Float32Array(data.audio);
    // ... append to buffer, compute new frames, store in melBuffer ...
  }

  if (data.type === 'get_features') {
    // Inference thread requests features — already computed!
    // Just slice from buffer and normalize. ~1-3ms.
    const { startFrame, endFrame } = data;
    const features = mel.normalize(melBuffer, totalFrames, endFrame - startFrame);
    self.postMessage({ features }, [features.buffer]);
  }
};
```

```js
// main thread — inference sees 0.0ms preprocessing
const melWorker = new Worker(new URL('./mel.worker.js', import.meta.url));

// Feed audio continuously as it arrives from microphone
audioEngine.onChunk(chunk => melWorker.postMessage({ type: 'push_audio', audio: chunk }));

// When inference needs features, they're already computed
const features = await requestFeaturesFromWorker(startFrame, endFrame);
model.transcribe(null, { precomputedFeatures: features }); // Preprocess: 0.0ms
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

### Raw Processing Time

Benchmarks on a modern desktop (Node.js):

| Duration | Processing Time | Realtime Factor |
|----------|----------------|-----------------|
| 0.5s | ~3ms | ~160x |
| 1s | ~7ms | ~140x |
| 5s | ~37ms | ~135x |
| 10s | ~70ms | ~140x |

### Incremental Caching (the real win)

In streaming ASR, you typically process overlapping windows (e.g. 5s window every 1s). With a stateless ONNX preprocessor, you recompute the full window every time. meljs caches prefix frames and only computes new audio:

| Mode | Time (5s window) | Speedup |
|------|-----------------|---------|
| Full recompute (or ONNX) | 71.7ms | — |
| Incremental (70% cached) | 36.6ms | **~2x** |

### Background Worker Pipeline (zero-latency preprocessing)

The biggest impact comes from running meljs in a dedicated Web Worker as a continuous mel producer. Because features are pre-computed before inference requests them, the inference thread sees:

| Architecture | Preprocessing Latency |
|-------------|----------------------|
| ONNX preprocessor (synchronous) | **~180ms** per 5s window |
| meljs in background worker | **0.0ms** (pre-computed, just a buffer read) |

This was measured in production with [boncukjs](https://github.com/ysdede/boncukjs) — a real-time transcription app. The mel worker continuously ingests audio chunks (~0.5ms per 80ms chunk), and when the encoder needs a 5s feature window, it's retrieved from the buffer in ~1-3ms. The inference pipeline reports `Preprocess: 0.0ms`.

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
