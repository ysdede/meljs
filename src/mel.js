/**
 * meljs — Pure JavaScript mel spectrogram and audio feature extraction.
 *
 * Zero dependencies. Works in browsers (including Web Workers) and Node.js.
 * NeMo-compatible pipeline: pre-emphasis → STFT → mel filterbank → log → normalize.
 *
 * Pipeline (matching NeMo / onnx-asr nemo.py exactly):
 *   1. Pre-emphasis:       x[n] = x[n] - coeff * x[n-1]  (Float32)
 *   2. Zero-pad:           N_FFT/2 samples on each side
 *   3. STFT:               Cast to Float64, symmetric Hann window, N_FFT-point FFT
 *   4. Power spectrum:     |real|² + |imag|²  → Float32
 *   5. Mel filterbank:     MatMul with Slaney-normalized triangular filterbank
 *   6. Log:                log(mel + guard)
 *   7. Normalize:          Per-feature mean/variance (Bessel-corrected, N-1 denominator)
 *
 * Default parameters match NeMo/Parakeet TDT models:
 *   sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
 *   preemph=0.97, mel_scale="slaney", norm="slaney", log_zero_guard=2^-24
 *
 * @module meljs
 */

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/** Default mel spectrogram parameters (NeMo-compatible). */
export const MEL_CONSTANTS = Object.freeze({
  SAMPLE_RATE: 16000,
  N_FFT: 512,
  WIN_LENGTH: 400,
  HOP_LENGTH: 160,
  PREEMPH: 0.97,
  LOG_ZERO_GUARD: 2 ** -24,           // ≈ 5.96e-8
  N_FREQ_BINS: (512 >> 1) + 1,       // 257
  DEFAULT_N_MELS: 128,
});

// Internal constants
const { SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP_LENGTH, PREEMPH, LOG_ZERO_GUARD, N_FREQ_BINS } = MEL_CONSTANTS;

// ═══════════════════════════════════════════════════════════════════════════════
// Slaney Mel Scale (matching torchaudio.functional.melscale_fbanks)
// ═══════════════════════════════════════════════════════════════════════════════

const F_SP = 200.0 / 3;                       // ~66.667 Hz spacing in linear region
const MIN_LOG_HZ = 1000.0;                    // transition from linear to log
const MIN_LOG_MEL = MIN_LOG_HZ / F_SP;        // = 15.0
const LOG_STEP = Math.log(6.4) / 27.0;        // step size in log region

/**
 * Convert frequency in Hz to Slaney mel scale.
 * Linear below 1000 Hz, logarithmic above.
 *
 * @param {number} freq - Frequency in Hz
 * @returns {number} Mel value
 */
export function hzToMel(freq) {
  return freq >= MIN_LOG_HZ
    ? MIN_LOG_MEL + Math.log(freq / MIN_LOG_HZ) / LOG_STEP
    : freq / F_SP;
}

/**
 * Convert Slaney mel value back to Hz.
 *
 * @param {number} mel - Mel value
 * @returns {number} Frequency in Hz
 */
export function melToHz(mel) {
  return mel >= MIN_LOG_MEL
    ? MIN_LOG_HZ * Math.exp(LOG_STEP * (mel - MIN_LOG_MEL))
    : mel * F_SP;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mel Filterbank
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a Slaney-normalized mel filterbank matrix.
 * Matches `torchaudio.functional.melscale_fbanks(norm="slaney", mel_scale="slaney")`.
 *
 * Layout: [nMels × N_FREQ_BINS] row-major (transposed from torchaudio's [257, nMels])
 * for cache-friendly access during per-frame matmul.
 *
 * @param {number} [nMels=128] - Number of mel bins (typically 80 or 128)
 * @param {number} [sampleRate=16000] - Audio sample rate
 * @param {number} [nFft=512] - FFT size
 * @returns {Float32Array} Filterbank matrix [nMels × nFreqBins]
 */
export function createMelFilterbank(nMels = 128, sampleRate = SAMPLE_RATE, nFft = N_FFT) {
  const nFreqBins = (nFft >> 1) + 1;
  const fMax = sampleRate / 2;

  // Linearly spaced frequency bins
  const allFreqs = new Float64Array(nFreqBins);
  for (let i = 0; i < nFreqBins; i++) {
    allFreqs[i] = (fMax * i) / (nFreqBins - 1);
  }

  // Mel-spaced center frequencies (nMels + 2 points)
  const melMin = hzToMel(0);
  const melMax = hzToMel(fMax);
  const nPoints = nMels + 2;
  const fPts = new Float64Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    fPts[i] = melToHz(melMin + ((melMax - melMin) * i) / (nPoints - 1));
  }

  // Differences between consecutive points
  const fDiff = new Float64Array(nPoints - 1);
  for (let i = 0; i < nPoints - 1; i++) {
    fDiff[i] = fPts[i + 1] - fPts[i];
  }

  // Build triangular filterbank with Slaney normalization
  const fb = new Float32Array(nMels * nFreqBins);
  for (let m = 0; m < nMels; m++) {
    const enorm = 2.0 / (fPts[m + 2] - fPts[m]);
    const fbOffset = m * nFreqBins;
    for (let k = 0; k < nFreqBins; k++) {
      const downSlope = (allFreqs[k] - fPts[m]) / fDiff[m];
      const upSlope = (fPts[m + 2] - allFreqs[k]) / fDiff[m + 1];
      fb[fbOffset + k] = Math.max(0, Math.min(downSlope, upSlope)) * enorm;
    }
  }

  return fb;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hann Window
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a symmetric Hann window zero-padded to nFft length.
 * Uses Float64Array for precision (matches ONNX's DOUBLE STFT).
 *
 * @param {number} [winLength=400] - Window length in samples
 * @param {number} [nFft=512] - FFT size (window is centered and zero-padded)
 * @returns {Float64Array} Padded Hann window of length nFft
 */
export function createPaddedHannWindow(winLength = WIN_LENGTH, nFft = N_FFT) {
  const window = new Float64Array(nFft);
  const padLeft = (nFft - winLength) >> 1;

  for (let n = 0; n < winLength; n++) {
    window[padLeft + n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (winLength - 1)));
  }

  return window;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFT (Radix-2 Cooley-Tukey, in-place)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Precompute twiddle factors for N-point FFT.
 *
 * @param {number} N - FFT size (must be power of 2)
 * @returns {{cos: Float64Array, sin: Float64Array}} Twiddle factors
 */
export function precomputeTwiddles(N) {
  const half = N >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  for (let i = 0; i < half; i++) {
    const angle = (-2 * Math.PI * i) / N;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }
  return { cos, sin };
}

/**
 * In-place radix-2 Cooley-Tukey FFT.
 *
 * @param {Float64Array} re - Real part (modified in-place)
 * @param {Float64Array} im - Imaginary part (modified in-place)
 * @param {number} N - FFT size (must be power of 2)
 * @param {{cos: Float64Array, sin: Float64Array}} tw - Precomputed twiddle factors
 */
export function fft(re, im, N, tw) {
  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < N - 1; i++) {
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
    let k = N >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }

  // Butterfly stages
  for (let len = 2; len <= N; len <<= 1) {
    const halfLen = len >> 1;
    const step = N / len;
    for (let i = 0; i < N; i += len) {
      for (let k = 0; k < halfLen; k++) {
        const twIdx = k * step;
        const wCos = tw.cos[twIdx];
        const wSin = tw.sin[twIdx];
        const p = i + k;
        const q = p + halfLen;
        const tRe = re[q] * wCos - im[q] * wSin;
        const tIm = re[q] * wSin + im[q] * wCos;
        re[q] = re[p] - tRe;
        im[q] = im[p] - tIm;
        re[p] += tRe;
        im[p] += tIm;
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MelSpectrogram — Full pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Pure JS mel spectrogram processor. Produces normalized log-mel features
 * identical to NeMo's nemo128.onnx / nemo80.onnx preprocessor.
 *
 * @example
 * ```js
 * import { MelSpectrogram } from 'meljs';
 *
 * const mel = new MelSpectrogram({ nMels: 128 });
 * const { features, length } = mel.process(audioFloat32Array);
 * // features: Float32Array [nMels × nFrames], normalized
 * // length: number of valid frames
 * ```
 */
export class MelSpectrogram {
  /**
   * @param {Object} [opts]
   * @param {number} [opts.nMels=128] - Number of mel bins (typically 80 or 128)
   * @param {number} [opts.sampleRate=16000] - Audio sample rate in Hz
   * @param {number} [opts.nFft=512] - FFT size
   * @param {number} [opts.winLength=400] - Window length in samples
   * @param {number} [opts.hopLength=160] - Hop length in samples
   * @param {number} [opts.preemph=0.97] - Pre-emphasis coefficient (0 to disable)
   * @param {number} [opts.logZeroGuard=2**-24] - Guard value for log computation
   */
  constructor(opts = {}) {
    this.nMels = opts.nMels || 128;
    this.sampleRate = opts.sampleRate || SAMPLE_RATE;
    this.nFft = opts.nFft || N_FFT;
    this.winLength = opts.winLength || WIN_LENGTH;
    this.hopLength = opts.hopLength || HOP_LENGTH;
    this.preemph = opts.preemph ?? PREEMPH;
    this.logZeroGuard = opts.logZeroGuard ?? LOG_ZERO_GUARD;
    this.nFreqBins = (this.nFft >> 1) + 1;

    // Precompute constants
    this.melFilterbank = createMelFilterbank(this.nMels, this.sampleRate, this.nFft);
    this.hannWindow = createPaddedHannWindow(this.winLength, this.nFft);
    this.twiddles = precomputeTwiddles(this.nFft);

    // Precompute sparse filterbank indices to avoid multiplying by zero
    this._fbStart = new Int32Array(this.nMels);
    this._fbEnd = new Int32Array(this.nMels);
    for (let m = 0; m < this.nMels; m++) {
      let start = -1;
      let end = -1;
      const offset = m * this.nFreqBins;
      for (let k = 0; k < this.nFreqBins; k++) {
        if (this.melFilterbank[offset + k] > 0) {
          if (start === -1) start = k;
          end = k + 1;
        }
      }
      this._fbStart[m] = start === -1 ? 0 : start;
      this._fbEnd[m] = end === -1 ? 0 : end;
    }

    // Pre-allocate reusable buffers
    this._fftRe = new Float64Array(this.nFft);
    this._fftIm = new Float64Array(this.nFft);
    this._powerBuf = new Float32Array(this.nFreqBins);
  }

  /**
   * Convert PCM audio to normalized log-mel spectrogram features.
   *
   * @param {Float32Array} audio - Mono PCM [-1,1] at the configured sample rate
   * @returns {{features: Float32Array, length: number}}
   *   - features: [nMels × nFrames] row-major, normalized (zero-mean, unit-variance)
   *   - length: number of valid frames
   */
  process(audio) {
    const { rawMel, nFrames, featuresLen } = this.computeRawMel(audio);
    if (featuresLen === 0) return { features: new Float32Array(0), length: 0 };
    const features = this.normalize(rawMel, nFrames, featuresLen);
    return { features, length: featuresLen };
  }

  /**
   * Compute raw (un-normalized) log-mel features.
   * Useful for incremental/streaming workflows where normalization
   * happens over a different window.
   *
   * @param {Float32Array} audio - Mono PCM audio
   * @returns {{rawMel: Float32Array, nFrames: number, featuresLen: number}}
   */
  computeRawMel(audio) {
    const N = audio.length;
    if (N === 0) return { rawMel: new Float32Array(0), nFrames: 0, featuresLen: 0 };

    // 1. Pre-emphasis
    const preemph = new Float32Array(N);
    preemph[0] = audio[0];
    for (let i = 1; i < N; i++) {
      preemph[i] = audio[i] - this.preemph * audio[i - 1];
    }

    // 2. Zero-pad: nFft/2 on each side
    const pad = this.nFft >> 1;
    const paddedLen = N + 2 * pad;
    const padded = new Float64Array(paddedLen);
    for (let i = 0; i < N; i++) padded[pad + i] = preemph[i];

    // 3. Frame counts
    const nFrames = Math.floor((paddedLen - this.nFft) / this.hopLength) + 1;
    const featuresLen = Math.floor(N / this.hopLength);
    if (featuresLen === 0) return { rawMel: new Float32Array(0), nFrames: 0, featuresLen: 0 };

    // 4. STFT + Power + Mel + Log
    const rawMel = new Float32Array(this.nMels * nFrames);
    const { _fftRe: fftRe, _fftIm: fftIm, _powerBuf: powerBuf } = this;
    const { hannWindow: window, melFilterbank: fb, nMels, twiddles: tw, nFft, nFreqBins, hopLength, logZeroGuard, _fbStart: fbStart, _fbEnd: fbEnd } = this;

    for (let t = 0; t < nFrames; t++) {
      const offset = t * hopLength;
      for (let k = 0; k < nFft; k++) { fftRe[k] = padded[offset + k] * window[k]; fftIm[k] = 0; }
      fft(fftRe, fftIm, nFft, tw);
      for (let k = 0; k < nFreqBins; k++) { powerBuf[k] = fftRe[k] * fftRe[k] + fftIm[k] * fftIm[k]; }
      for (let m = 0; m < nMels; m++) {
        let melVal = 0;
        const fbOff = m * nFreqBins;
        const start = fbStart[m];
        const end = fbEnd[m];
        for (let k = start; k < end; k++) melVal += powerBuf[k] * fb[fbOff + k];
        rawMel[m * nFrames + t] = Math.log(melVal + logZeroGuard);
      }
    }

    return { rawMel, nFrames, featuresLen };
  }

  /**
   * Apply per-feature normalization (Bessel-corrected) to raw mel features.
   * Produces zero-mean, unit-variance features per mel bin.
   *
   * @param {Float32Array} rawMel - Pre-normalization features [nMels, nFrames]
   * @param {number} nFrames - Total number of frames
   * @param {number} featuresLen - Number of valid frames to normalize over
   * @returns {Float32Array} Normalized features (new array)
   */
  normalize(rawMel, nFrames, featuresLen) {
    const { nMels } = this;
    const features = new Float32Array(nMels * featuresLen);

    for (let m = 0; m < nMels; m++) {
      const srcBase = m * nFrames;
      const dstBase = m * featuresLen;
      let sum = 0;
      for (let t = 0; t < featuresLen; t++) sum += rawMel[srcBase + t];
      const mean = sum / featuresLen;

      let varSum = 0;
      for (let t = 0; t < featuresLen; t++) { const d = rawMel[srcBase + t] - mean; varSum += d * d; }
      const invStd = featuresLen > 1 ? 1.0 / (Math.sqrt(varSum / (featuresLen - 1)) + 1e-5) : 0;

      for (let t = 0; t < featuresLen; t++) features[dstBase + t] = (rawMel[srcBase + t] - mean) * invStd;
    }

    return features;
  }

  /**
   * Convert a sample count to a frame count.
   * @param {number} samples - Number of audio samples
   * @returns {number} Number of frames
   */
  samplesToFrames(samples) {
    return Math.floor(samples / this.hopLength);
  }

  /**
   * Convert a frame count to approximate sample count.
   * @param {number} frames - Number of frames
   * @returns {number} Number of audio samples
   */
  framesToSamples(frames) {
    return frames * this.hopLength;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IncrementalMelSpectrogram — Streaming with caching
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Incrementally computes mel features for streaming with overlapping windows.
 * Caches raw (pre-normalization) mel frames from the overlap prefix,
 * reuses them on the next call, and only computes new frames.
 *
 * @example
 * ```js
 * import { IncrementalMelSpectrogram } from 'meljs';
 *
 * const mel = new IncrementalMelSpectrogram({ nMels: 128 });
 *
 * // First chunk: full computation
 * const r1 = mel.process(audioWindow1, 0);
 *
 * // Second chunk: 70% overlap → reuses ~70% of frames
 * const overlapSamples = Math.floor(audioWindow2.length * 0.7);
 * const r2 = mel.process(audioWindow2, overlapSamples);
 * console.log(r2.cachedFrames, r2.newFrames); // e.g., 347, 153
 *
 * // New recording
 * mel.reset();
 * ```
 */
export class IncrementalMelSpectrogram {
  /**
   * @param {Object} [opts] - Same options as MelSpectrogram, plus:
   * @param {number} [opts.boundaryFrames=3] - Extra frames to recompute at boundary for safety
   */
  constructor(opts = {}) {
    this.processor = new MelSpectrogram(opts);
    this.nMels = this.processor.nMels;
    this.boundaryFrames = opts.boundaryFrames || 3;

    this._cachedRawMel = null;
    this._cachedNFrames = 0;
    this._cachedAudioLen = 0;
    this._cachedFeaturesLen = 0;
  }

  /**
   * Process audio with incremental mel computation.
   *
   * @param {Float32Array} audio - Full audio window
   * @param {number} [prefixSamples=0] - Number of leading samples identical to previous call
   * @returns {{features: Float32Array, length: number, cached: boolean, cachedFrames: number, newFrames: number}}
   */
  process(audio, prefixSamples = 0) {
    const N = audio.length;
    if (N === 0) return { features: new Float32Array(0), length: 0, cached: false, cachedFrames: 0, newFrames: 0 };

    const canReuse = prefixSamples > 0 && this._cachedRawMel !== null && prefixSamples <= this._cachedAudioLen;

    if (!canReuse) {
      const result = this.processor.process(audio);
      const { rawMel, nFrames, featuresLen } = this.processor.computeRawMel(audio);
      this._cachedRawMel = rawMel;
      this._cachedNFrames = nFrames;
      this._cachedAudioLen = N;
      this._cachedFeaturesLen = featuresLen;
      return { ...result, cached: false, cachedFrames: 0, newFrames: featuresLen };
    }

    // Incremental path
    const prefixFrames = Math.floor(prefixSamples / this.processor.hopLength);
    const safeFrames = Math.max(0, Math.min(prefixFrames - this.boundaryFrames, this._cachedFeaturesLen));

    const { rawMel, nFrames, featuresLen } = this.processor.computeRawMel(audio);

    if (safeFrames > 0 && this._cachedRawMel) {
      for (let m = 0; m < this.nMels; m++) {
        const srcBase = m * this._cachedNFrames;
        const dstBase = m * nFrames;
        for (let t = 0; t < safeFrames; t++) rawMel[dstBase + t] = this._cachedRawMel[srcBase + t];
      }
    }

    const features = this.processor.normalize(rawMel, nFrames, featuresLen);

    this._cachedRawMel = rawMel;
    this._cachedNFrames = nFrames;
    this._cachedAudioLen = N;
    this._cachedFeaturesLen = featuresLen;

    return { features, length: featuresLen, cached: true, cachedFrames: safeFrames, newFrames: featuresLen - safeFrames };
  }

  /** Reset the cache. Call when starting a new recording session. */
  reset() { this._cachedRawMel = null; this._cachedNFrames = 0; this._cachedAudioLen = 0; this._cachedFeaturesLen = 0; }

  /** Alias for reset(). */
  clear() { this.reset(); }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Legacy aliases (for parakeet.js compatibility)
// ═══════════════════════════════════════════════════════════════════════════════

/** @deprecated Use MelSpectrogram instead */
export const JsPreprocessor = MelSpectrogram;

/** @deprecated Use IncrementalMelSpectrogram instead */
export const IncrementalMelProcessor = IncrementalMelSpectrogram;
