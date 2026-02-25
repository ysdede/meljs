import {
  MEL_CONSTANTS,
  createMelFilterbank,
  createPaddedHannWindow,
  precomputeTwiddles,
  fft,
} from "./mel.js";

const {
  SAMPLE_RATE,
  N_FFT,
  WIN_LENGTH,
  HOP_LENGTH,
  PREEMPH,
  LOG_ZERO_GUARD,
  N_FREQ_BINS,
} = MEL_CONSTANTS;

const PARAKEET_FB_CACHE = new Map();
let PARAKEET_HANN_WINDOW = null;
let PARAKEET_TWIDDLES_FULL = null;
let PARAKEET_TWIDDLES_HALF = null;

function getCachedFilterbank(nMels) {
  let cached = PARAKEET_FB_CACHE.get(nMels);
  if (!cached) {
    cached = createMelFilterbank(nMels, SAMPLE_RATE, N_FFT);
    PARAKEET_FB_CACHE.set(nMels, cached);
  }
  return cached;
}

function getCachedHannWindow() {
  if (!PARAKEET_HANN_WINDOW) {
    PARAKEET_HANN_WINDOW = createPaddedHannWindow(WIN_LENGTH, N_FFT);
  }
  return PARAKEET_HANN_WINDOW;
}

function getTwiddlesFull() {
  if (!PARAKEET_TWIDDLES_FULL) {
    PARAKEET_TWIDDLES_FULL = precomputeTwiddles(N_FFT);
  }
  return PARAKEET_TWIDDLES_FULL;
}

function getTwiddlesHalf() {
  if (!PARAKEET_TWIDDLES_HALF) {
    PARAKEET_TWIDDLES_HALF = precomputeTwiddles(N_FFT >> 1);
  }
  return PARAKEET_TWIDDLES_HALF;
}

function validateParakeetDefaults(opts = {}) {
  const checks = [
    ["sampleRate", SAMPLE_RATE],
    ["nFft", N_FFT],
    ["winLength", WIN_LENGTH],
    ["hopLength", HOP_LENGTH],
    ["preemph", PREEMPH],
  ];
  for (const [key, expected] of checks) {
    if (opts[key] != null && opts[key] !== expected) {
      throw new Error(
        `Parakeet variants use fixed ${key}=${expected}. Received ${opts[key]}. ` +
          `Use MelSpectrogram for fully custom parameters.`
      );
    }
  }
}

function normalizeVariantName(name) {
  const n = String(name || "pr75").toLowerCase();
  if (n === "current" || n === "baseline" || n === "master") return "current";
  if (n === "pr74" || n === "pr-74" || n === "candidate74" || n === "realfft74") return "pr74";
  if (n === "pr75" || n === "pr-75" || n === "candidate75" || n === "realfft75") return "pr75";
  if (n === "pr84" || n === "pr-84" || n === "candidate84" || n === "realfft84") return "pr84";
  throw new Error(`Unknown variant "${name}". Expected one of: current, pr74, pr75, pr84`);
}

class ParakeetMelProcessorBase {
  constructor(opts = {}, variantName = "unknown") {
    validateParakeetDefaults(opts);

    this.variant = variantName;
    this.nMels = opts.nMels || 128;
    this.sampleRate = SAMPLE_RATE;
    this.nFft = N_FFT;
    this.winLength = WIN_LENGTH;
    this.hopLength = HOP_LENGTH;
    this.preemph = PREEMPH;
    this.logZeroGuard = LOG_ZERO_GUARD;
    this.nFreqBins = N_FREQ_BINS;

    this.melFilterbank = getCachedFilterbank(this.nMels);
    this.hannWindow = getCachedHannWindow();
    this.fbBounds = new Int32Array(this.nMels * 2);
    for (let m = 0; m < this.nMels; m++) {
      const fbOff = m * N_FREQ_BINS;
      let start = -1;
      let end = -1;
      for (let k = 0; k < N_FREQ_BINS; k++) {
        if (this.melFilterbank[fbOff + k] > 0) {
          if (start === -1) start = k;
          end = k;
        }
      }
      if (start === -1) {
        start = 0;
        end = -1;
      }
      this.fbBounds[m * 2] = start;
      this.fbBounds[m * 2 + 1] = end + 1;
    }

    this._powerBuf = new Float32Array(N_FREQ_BINS);
    this._paddedBuffer = null;
    this._initVariantBuffers();
  }

  _initVariantBuffers() {
    throw new Error("_initVariantBuffers() must be implemented by subclasses");
  }

  _computePowerSpectrumFrame(_padded, _offset, _powerBuf) {
    throw new Error("_computePowerSpectrumFrame() must be implemented by subclasses");
  }

  process(audio) {
    const { rawMel, nFrames, featuresLen } = this.computeRawMel(audio);
    if (featuresLen === 0) {
      return { features: new Float32Array(0), length: 0 };
    }
    return {
      features: this.normalize(rawMel, nFrames, featuresLen),
      length: featuresLen,
    };
  }

  computeRawMel(audio, startFrame = 0, outBuffer = null) {
    const N = audio.length;
    if (N === 0) {
      return {
        rawMel: outBuffer ? outBuffer.subarray(0, 0) : new Float32Array(0),
        nFrames: 0,
        featuresLen: 0,
      };
    }

    const pad = N_FFT >> 1;
    const paddedLen = N + 2 * pad;
    let paddedWasReallocated = false;
    if (!this._paddedBuffer || this._paddedBuffer.length < paddedLen) {
      const newSize = Math.ceil(paddedLen * 1.2);
      this._paddedBuffer = new Float64Array(newSize);
      paddedWasReallocated = true;
    }
    const padded = this._paddedBuffer;

    padded[pad] = Math.fround(audio[0]);
    for (let i = 1; i < N; i++) {
      padded[pad + i] = Math.fround(audio[i] - PREEMPH * audio[i - 1]);
    }

    if (!paddedWasReallocated) {
      padded.fill(0, pad + N, paddedLen);
    }

    const nFrames = Math.floor((paddedLen - N_FFT) / HOP_LENGTH) + 1;
    const featuresLen = Math.floor(N / HOP_LENGTH);
    if (featuresLen === 0) {
      return { rawMel: new Float32Array(0), nFrames: 0, featuresLen: 0 };
    }

    const reqSize = this.nMels * nFrames;
    let rawMel;
    if (outBuffer && outBuffer.length >= reqSize) {
      rawMel = outBuffer.subarray(0, reqSize);
      if (startFrame > 0) {
        for (let m = 0; m < this.nMels; m++) {
          rawMel.fill(0, m * nFrames, m * nFrames + startFrame);
        }
      }
    } else {
      rawMel = new Float32Array(reqSize);
    }

    const powerBuf = this._powerBuf;
    const fb = this.melFilterbank;
    const fbBounds = this.fbBounds;
    const nMels = this.nMels;

    for (let t = startFrame; t < nFrames; t++) {
      const offset = t * HOP_LENGTH;
      this._computePowerSpectrumFrame(padded, offset, powerBuf);

      for (let m = 0; m < nMels; m++) {
        let melVal = 0;
        const fbOff = m * N_FREQ_BINS;
        const start = fbBounds[m * 2];
        const end = fbBounds[m * 2 + 1];
        for (let k = start; k < end; k++) {
          melVal += powerBuf[k] * fb[fbOff + k];
        }
        rawMel[m * nFrames + t] = Math.log(melVal + LOG_ZERO_GUARD);
      }
    }

    return { rawMel, nFrames, featuresLen };
  }

  normalize(rawMel, nFrames, featuresLen, outBuffer = null) {
    const nMels = this.nMels;
    const reqSize = nMels * featuresLen;
    let features;

    if (outBuffer && outBuffer.length >= reqSize) {
      features = outBuffer.subarray(0, reqSize);
    } else {
      features = new Float32Array(reqSize);
    }

    for (let m = 0; m < nMels; m++) {
      const srcBase = m * nFrames;
      const dstBase = m * featuresLen;

      let sum = 0;
      for (let t = 0; t < featuresLen; t++) {
        sum += rawMel[srcBase + t];
      }
      const mean = sum / featuresLen;

      let varSum = 0;
      for (let t = 0; t < featuresLen; t++) {
        const d = rawMel[srcBase + t] - mean;
        varSum += d * d;
      }
      const invStd =
        featuresLen > 1
          ? 1.0 / (Math.sqrt(varSum / (featuresLen - 1)) + 1e-5)
          : 0;

      for (let t = 0; t < featuresLen; t++) {
        features[dstBase + t] = (rawMel[srcBase + t] - mean) * invStd;
      }
    }

    return features;
  }
}

export class ParakeetCurrentMelProcessor extends ParakeetMelProcessorBase {
  constructor(opts = {}) {
    super(opts, "current");
  }

  _initVariantBuffers() {
    this._twiddles = getTwiddlesFull();
    this._fftRe = new Float64Array(N_FFT);
    this._fftIm = new Float64Array(N_FFT);
  }

  _computePowerSpectrumFrame(padded, offset, powerBuf) {
    const fftRe = this._fftRe;
    const fftIm = this._fftIm;
    const window = this.hannWindow;
    const tw = this._twiddles;

    for (let k = 0; k < N_FFT; k++) {
      fftRe[k] = padded[offset + k] * window[k];
      fftIm[k] = 0;
    }
    fft(fftRe, fftIm, N_FFT, tw);
    for (let k = 0; k < N_FREQ_BINS; k++) {
      powerBuf[k] = fftRe[k] * fftRe[k] + fftIm[k] * fftIm[k];
    }
  }
}

export class ParakeetMelProcessorPr74 extends ParakeetMelProcessorBase {
  constructor(opts = {}) {
    super(opts, "pr74");
  }

  _initVariantBuffers() {
    this._twiddlesFull = getTwiddlesFull();
    this._twiddlesHalf = getTwiddlesHalf();
    this._fftRe = new Float64Array(N_FFT);
    this._fftIm = new Float64Array(N_FFT);
  }

  _computePowerSpectrumFrame(padded, offset, powerBuf) {
    const fftRe = this._fftRe;
    const fftIm = this._fftIm;
    const window = this.hannWindow;
    const twFull = this._twiddlesFull;
    const twHalf = this._twiddlesHalf;
    const halfN = N_FFT >> 1;
    const quarterN = halfN >> 1;

    for (let k = 0; k < halfN; k++) {
      const idx = k << 1;
      fftRe[k] = padded[offset + idx] * window[idx];
      fftIm[k] = padded[offset + idx + 1] * window[idx + 1];
    }

    fft(fftRe, fftIm, halfN, twHalf);

    const z0r = fftRe[0];
    const z0i = fftIm[0];
    powerBuf[0] = (z0r + z0i) * (z0r + z0i);
    powerBuf[halfN] = (z0r - z0i) * (z0r - z0i);

    for (let k = 1; k < quarterN; k++) {
      const rk = fftRe[k];
      const ik = fftIm[k];
      const rnk = fftRe[halfN - k];
      const ink = fftIm[halfN - k];

      const xeR = 0.5 * (rk + rnk);
      const xeI = 0.5 * (ik - ink);
      const xoR = 0.5 * (ik + ink);
      const xoI = -0.5 * (rk - rnk);

      const wc = twFull.cos[k];
      const ws = twFull.sin[k];
      const tr = xoR * wc - xoI * ws;
      const ti = xoR * ws + xoI * wc;

      const xkR = xeR + tr;
      const xkI = xeI + ti;
      powerBuf[k] = xkR * xkR + xkI * xkI;

      const xnkR = xeR - tr;
      const xnkI = xeI - ti;
      powerBuf[halfN - k] = xnkR * xnkR + xnkI * xnkI;
    }

    const rQuarter = fftRe[quarterN];
    const iQuarter = fftIm[quarterN];
    powerBuf[quarterN] = rQuarter * rQuarter + iQuarter * iQuarter;
  }
}

export class ParakeetMelProcessorPr75 extends ParakeetMelProcessorBase {
  constructor(opts = {}) {
    super(opts, "pr75");
  }

  _initVariantBuffers() {
    this._twiddlesFull = getTwiddlesFull();
    this._twiddlesHalf = getTwiddlesHalf();
    this._fftRe = new Float64Array(N_FFT >> 1);
    this._fftIm = new Float64Array(N_FFT >> 1);
  }

  _computePowerSpectrumFrame(padded, offset, powerBuf) {
    const fftRe = this._fftRe;
    const fftIm = this._fftIm;
    const window = this.hannWindow;
    const twFull = this._twiddlesFull;
    const twHalf = this._twiddlesHalf;
    const N2 = N_FFT >> 1;
    const N4 = N2 >> 1;

    for (let k = 0; k < N2; k++) {
      const idx = offset + (k << 1);
      fftRe[k] = padded[idx] * window[k << 1];
      fftIm[k] = padded[idx + 1] * window[(k << 1) + 1];
    }

    fft(fftRe, fftIm, N2, twHalf);

    const z0Re = fftRe[0];
    const z0Im = fftIm[0];
    powerBuf[0] = (z0Re + z0Im) * (z0Re + z0Im);
    powerBuf[N2] = (z0Re - z0Im) * (z0Re - z0Im);

    for (let k = 1; k <= N4; k++) {
      const kRev = N2 - k;
      const reK = fftRe[k];
      const imK = fftIm[k];
      const reRev = k === N4 ? reK : fftRe[kRev];
      const imRev = k === N4 ? imK : fftIm[kRev];

      const reA = 0.5 * (reK + reRev);
      const imA = 0.5 * (imK - imRev);
      const reC = 0.5 * (imK + imRev);
      const imC = -0.5 * (reK - reRev);

      const c = twFull.cos[k];
      const s = twFull.sin[k];
      const reCW = reC * c - imC * s;
      const imCW = reC * s + imC * c;

      const xRe = reA + reCW;
      const xIm = imA + imCW;
      powerBuf[k] = xRe * xRe + xIm * xIm;

      if (k < N4) {
        const x2Re = reA - reCW;
        const x2Im = imA - imCW;
        powerBuf[kRev] = x2Re * x2Re + x2Im * x2Im;
      }
    }
  }
}

export class ParakeetMelProcessorPr84 extends ParakeetMelProcessorPr75 {
  constructor(opts = {}) {
    super(opts);
    this.variant = "pr84";
  }
}

export const PARAKEET_MEL_VARIANTS = Object.freeze([
  "current",
  "pr74",
  "pr75",
  "pr84",
]);

export function createParakeetMelProcessor(opts = {}) {
  const variant = normalizeVariantName(opts.variant || "pr75");
  const sharedOpts = { ...opts };
  delete sharedOpts.variant;

  switch (variant) {
    case "current":
      return new ParakeetCurrentMelProcessor(sharedOpts);
    case "pr74":
      return new ParakeetMelProcessorPr74(sharedOpts);
    case "pr75":
      return new ParakeetMelProcessorPr75(sharedOpts);
    case "pr84":
      return new ParakeetMelProcessorPr84(sharedOpts);
    default:
      throw new Error(`Unhandled variant: ${variant}`);
  }
}

export class ParakeetIncrementalMelProcessor {
  constructor(opts = {}) {
    this.variant = normalizeVariantName(opts.variant || "pr75");
    this.boundaryFrames = opts.boundaryFrames || 3;

    const melOpts = { ...opts };
    delete melOpts.boundaryFrames;
    this.preprocessor = createParakeetMelProcessor({
      ...melOpts,
      variant: this.variant,
    });
    this.nMels = this.preprocessor.nMels;

    this._cachedRawMel = null;
    this._cachedNFrames = 0;
    this._cachedAudioLen = 0;
    this._cachedFeaturesLen = 0;
  }

  process(audio, prefixSamples = 0) {
    const N = audio.length;
    if (N === 0) {
      return {
        features: new Float32Array(0),
        length: 0,
        cached: false,
        cachedFrames: 0,
        newFrames: 0,
      };
    }

    const canReuse =
      prefixSamples > 0 &&
      this._cachedRawMel !== null &&
      prefixSamples <= this._cachedAudioLen;

    if (!canReuse) {
      const raw = this.preprocessor.computeRawMel(audio);
      const features = this.preprocessor.normalize(
        raw.rawMel,
        raw.nFrames,
        raw.featuresLen
      );
      this._cachedRawMel = raw.rawMel;
      this._cachedNFrames = raw.nFrames;
      this._cachedAudioLen = N;
      this._cachedFeaturesLen = raw.featuresLen;
      return {
        features,
        length: raw.featuresLen,
        cached: false,
        cachedFrames: 0,
        newFrames: raw.featuresLen,
      };
    }

    const prefixFrames = Math.floor(prefixSamples / HOP_LENGTH);
    const safeFrames = Math.max(
      0,
      Math.min(prefixFrames - this.boundaryFrames, this._cachedFeaturesLen)
    );

    const raw = this.preprocessor.computeRawMel(audio, safeFrames);
    const copyFrames = Math.min(safeFrames, raw.featuresLen);

    if (copyFrames > 0 && this._cachedRawMel) {
      for (let m = 0; m < this.nMels; m++) {
        const srcBase = m * this._cachedNFrames;
        const dstBase = m * raw.nFrames;
        for (let t = 0; t < copyFrames; t++) {
          raw.rawMel[dstBase + t] = this._cachedRawMel[srcBase + t];
        }
      }
    }

    const features = this.preprocessor.normalize(
      raw.rawMel,
      raw.nFrames,
      raw.featuresLen
    );

    this._cachedRawMel = raw.rawMel;
    this._cachedNFrames = raw.nFrames;
    this._cachedAudioLen = N;
    this._cachedFeaturesLen = raw.featuresLen;

    return {
      features,
      length: raw.featuresLen,
      cached: true,
      cachedFrames: copyFrames,
      newFrames: raw.featuresLen - copyFrames,
    };
  }

  reset() {
    this._cachedRawMel = null;
    this._cachedNFrames = 0;
    this._cachedAudioLen = 0;
    this._cachedFeaturesLen = 0;
  }

  clear() {
    this.reset();
  }
}

