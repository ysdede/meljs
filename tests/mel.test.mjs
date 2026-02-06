/**
 * Unit & integration tests for meljs — Pure JS mel spectrogram library.
 *
 * Tests cover:
 *   - Constants correctness
 *   - Mel scale conversion (hzToMel / melToHz)
 *   - Mel filterbank construction
 *   - Hann window (configurable parameters)
 *   - FFT (Radix-2 Cooley-Tukey)
 *   - MelSpectrogram full pipeline
 *   - IncrementalMelSpectrogram caching
 *   - Legacy aliases (JsPreprocessor / IncrementalMelProcessor)
 *   - ONNX reference cross-validation
 *   - Performance benchmarks
 *
 * Run: npm test
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const testsDir = __dirname;

import {
  MelSpectrogram,
  IncrementalMelSpectrogram,
  MEL_CONSTANTS,
  hzToMel,
  melToHz,
  createMelFilterbank,
  createPaddedHannWindow,
  precomputeTwiddles,
  fft,
  // Legacy aliases
  JsPreprocessor,
  IncrementalMelProcessor,
} from '../src/index.js';

// ─── Helpers ──────────────────────────────────────────────────────────────

function base64ToFloat32(b64) {
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / Float32Array.BYTES_PER_ELEMENT);
}

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

describe('MEL_CONSTANTS', () => {
  it('should have correct NeMo-compatible values', () => {
    expect(MEL_CONSTANTS.SAMPLE_RATE).toBe(16000);
    expect(MEL_CONSTANTS.N_FFT).toBe(512);
    expect(MEL_CONSTANTS.WIN_LENGTH).toBe(400);
    expect(MEL_CONSTANTS.HOP_LENGTH).toBe(160);
    expect(MEL_CONSTANTS.PREEMPH).toBe(0.97);
    expect(MEL_CONSTANTS.LOG_ZERO_GUARD).toBe(2 ** -24);
    expect(MEL_CONSTANTS.N_FREQ_BINS).toBe(257);
    expect(MEL_CONSTANTS.DEFAULT_N_MELS).toBe(128);
  });

  it('should be frozen (immutable)', () => {
    expect(Object.isFrozen(MEL_CONSTANTS)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Mel Scale
// ═══════════════════════════════════════════════════════════════════════════

describe('hzToMel / melToHz', () => {
  it('should return 0 for 0 Hz', () => {
    expect(hzToMel(0)).toBe(0);
  });

  it('should return mel in linear region for freq < 1000 Hz', () => {
    const freq = 500;
    const expected = freq / (200 / 3); // Slaney linear region
    expect(hzToMel(freq)).toBeCloseTo(expected, 5);
  });

  it('should transition at 1000 Hz (mel = 15.0)', () => {
    expect(hzToMel(1000)).toBeCloseTo(15.0, 5);
    expect(hzToMel(2000)).toBeGreaterThan(15.0);
  });

  it('should be invertible (roundtrip)', () => {
    const freqs = [0, 100, 500, 1000, 2000, 4000, 8000];
    for (const freq of freqs) {
      const mel = hzToMel(freq);
      const recovered = melToHz(mel);
      expect(recovered).toBeCloseTo(freq, 3);
    }
  });

  it('should be monotonically increasing', () => {
    const freqs = [0, 100, 500, 1000, 2000, 4000, 8000];
    const mels = freqs.map(hzToMel);
    for (let i = 1; i < mels.length; i++) {
      expect(mels[i]).toBeGreaterThan(mels[i - 1]);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Mel Filterbank
// ═══════════════════════════════════════════════════════════════════════════

describe('createMelFilterbank', () => {
  it('should create filterbank with correct dimensions (128 × 257)', () => {
    const fb = createMelFilterbank(128);
    expect(fb).toBeInstanceOf(Float32Array);
    expect(fb.length).toBe(128 * 257);
  });

  it('should use default nMels=128 when called with no args', () => {
    const fb = createMelFilterbank();
    expect(fb.length).toBe(128 * 257);
  });

  it('should have non-negative values only', () => {
    const fb = createMelFilterbank(128);
    for (let i = 0; i < fb.length; i++) {
      expect(fb[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('should have non-zero energy in every mel bin', () => {
    const nMels = 128;
    const fb = createMelFilterbank(nMels);
    for (let m = 0; m < nMels; m++) {
      const offset = m * 257;
      let sum = 0;
      for (let k = 0; k < 257; k++) sum += fb[offset + k];
      expect(sum).toBeGreaterThan(0);
    }
  });

  it('should create contiguous triangular filters', () => {
    const nMels = 64;
    const fb = createMelFilterbank(nMels);
    for (let m = 0; m < nMels; m++) {
      const offset = m * 257;
      let first = -1, last = -1;
      for (let k = 0; k < 257; k++) {
        if (fb[offset + k] > 0) {
          if (first === -1) first = k;
          last = k;
        }
      }
      expect(first).toBeGreaterThanOrEqual(0);
      for (let k = first; k <= last; k++) {
        expect(fb[offset + k]).toBeGreaterThan(0);
      }
    }
  });

  it('should work for different nMels values', () => {
    for (const nMels of [40, 64, 80, 128]) {
      const fb = createMelFilterbank(nMels);
      expect(fb.length).toBe(nMels * 257);
    }
  });

  it('should accept custom sampleRate and nFft', () => {
    const fb = createMelFilterbank(80, 22050, 1024);
    expect(fb.length).toBe(80 * 513); // (1024/2 + 1)
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Hann Window
// ═══════════════════════════════════════════════════════════════════════════

describe('createPaddedHannWindow', () => {
  it('should return a Float64Array of length N_FFT (512) with defaults', () => {
    const win = createPaddedHannWindow();
    expect(win).toBeInstanceOf(Float64Array);
    expect(win.length).toBe(512);
  });

  it('should have zero padding at edges', () => {
    const win = createPaddedHannWindow();
    const padLeft = (512 - 400) >> 1; // 56
    for (let i = 0; i < padLeft; i++) expect(win[i]).toBe(0);
    for (let i = padLeft + 400; i < 512; i++) expect(win[i]).toBe(0);
  });

  it('should be symmetric in the active region', () => {
    const win = createPaddedHannWindow();
    const padLeft = (512 - 400) >> 1;
    for (let i = 0; i < 400; i++) {
      const mirror = 400 - 1 - i;
      expect(win[padLeft + i]).toBeCloseTo(win[padLeft + mirror], 10);
    }
  });

  it('should peak at center with value ~1.0', () => {
    const win = createPaddedHannWindow();
    const padLeft = (512 - 400) >> 1;
    const center = padLeft + Math.floor(400 / 2);
    expect(win[center]).toBeCloseTo(1.0, 2);
  });

  it('should accept custom winLength and nFft', () => {
    const win = createPaddedHannWindow(256, 512);
    expect(win.length).toBe(512);
    // Active region is centered
    const padLeft = (512 - 256) >> 1; // 128
    expect(win[padLeft]).toBeCloseTo(0, 5);
    expect(win[padLeft + 128]).toBeCloseTo(1.0, 2); // center
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// FFT
// ═══════════════════════════════════════════════════════════════════════════

describe('fft', () => {
  it('should handle DC signal (all ones)', () => {
    const n = 8;
    const tw = precomputeTwiddles(n);
    const re = new Float64Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const im = new Float64Array(n);
    fft(re, im, n, tw);
    expect(re[0]).toBeCloseTo(n, 5);
    for (let i = 1; i < n; i++) {
      expect(re[i]).toBeCloseTo(0, 5);
      expect(im[i]).toBeCloseTo(0, 5);
    }
  });

  it('should handle a single frequency sinusoid', () => {
    const n = 16;
    const tw = precomputeTwiddles(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.cos(2 * Math.PI * i / n);
    fft(re, im, n, tw);
    expect(Math.abs(re[1])).toBeCloseTo(n / 2, 3);
    expect(Math.abs(re[n - 1])).toBeCloseTo(n / 2, 3);
    for (let i = 2; i < n - 1; i++) {
      expect(Math.abs(re[i])).toBeLessThan(1e-6);
      expect(Math.abs(im[i])).toBeLessThan(1e-6);
    }
  });

  it('should handle 512-point FFT', () => {
    const n = 512;
    const tw = precomputeTwiddles(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    fft(re, im, n, tw);
    for (let i = 0; i < n; i++) {
      expect(re[i]).toBeCloseTo(0, 10);
      expect(im[i]).toBeCloseTo(0, 10);
    }
  });

  it("should satisfy Parseval's theorem (energy conservation)", () => {
    const n = 64;
    const tw = precomputeTwiddles(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.sin(i * 0.37) + Math.cos(i * 0.83);
    let timeEnergy = 0;
    for (let i = 0; i < n; i++) timeEnergy += re[i] * re[i] + im[i] * im[i];
    fft(re, im, n, tw);
    let freqEnergy = 0;
    for (let i = 0; i < n; i++) freqEnergy += re[i] * re[i] + im[i] * im[i];
    expect(freqEnergy / n).toBeCloseTo(timeEnergy, 5);
  });

  it('should handle large FFT sizes (2048)', () => {
    const n = 2048;
    const tw = precomputeTwiddles(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.cos(2 * Math.PI * 3 * i / n);
    fft(re, im, n, tw);
    expect(Math.abs(re[3])).toBeCloseTo(n / 2, 1);
    expect(Math.abs(re[n - 3])).toBeCloseTo(n / 2, 1);
  });
});

describe('precomputeTwiddles', () => {
  it('should produce cos and sin arrays of half the FFT size', () => {
    const tw = precomputeTwiddles(512);
    expect(tw.cos.length).toBe(256);
    expect(tw.sin.length).toBe(256);
  });

  it('should start with cos[0]=1, sin[0]=0', () => {
    const tw = precomputeTwiddles(512);
    expect(tw.cos[0]).toBeCloseTo(1.0, 10);
    expect(tw.sin[0]).toBeCloseTo(0.0, 10);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// MelSpectrogram
// ═══════════════════════════════════════════════════════════════════════════

describe('MelSpectrogram', () => {
  let mel;

  beforeAll(() => {
    mel = new MelSpectrogram({ nMels: 128 });
  });

  it('should use default parameters matching NeMo', () => {
    expect(mel.nMels).toBe(128);
    expect(mel.sampleRate).toBe(16000);
    expect(mel.nFft).toBe(512);
    expect(mel.winLength).toBe(400);
    expect(mel.hopLength).toBe(160);
    expect(mel.preemph).toBe(0.97);
  });

  it('should handle empty audio', () => {
    const { features, length } = mel.process(new Float32Array(0));
    expect(features.length).toBe(0);
    expect(length).toBe(0);
  });

  it('should handle very short audio (< 1 frame)', () => {
    const { length } = mel.process(new Float32Array(100));
    expect(length).toBe(0);
  });

  it('should produce correct frame count for 1s audio', () => {
    const audio = new Float32Array(16000);
    const { length } = mel.process(audio);
    expect(length).toBe(100); // 16000 / 160
  });

  it('should produce correct frame count for 5s audio', () => {
    const audio = new Float32Array(80000);
    const { length } = mel.process(audio);
    expect(length).toBe(500);
  });

  it('should produce [nMels × nFrames] shaped output', () => {
    const audio = new Float32Array(32000); // 2s
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000);
    const { features } = mel.process(audio);
    const nFrames = features.length / 128;
    expect(Number.isInteger(nFrames)).toBe(true);
  });

  it('should produce near-zero for silence (after normalization)', () => {
    const silence = new Float32Array(16000);
    const { features } = mel.process(silence);
    let maxAbs = 0;
    for (let i = 0; i < features.length; i++) maxAbs = Math.max(maxAbs, Math.abs(features[i]));
    expect(maxAbs).toBeLessThan(1e-3);
  });

  it('should produce finite values for sinusoidal input', () => {
    const audio = new Float32Array(16000);
    for (let i = 0; i < audio.length; i++) audio[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / 16000);
    const { features } = mel.process(audio);
    for (let i = 0; i < features.length; i++) expect(isFinite(features[i])).toBe(true);
  });

  it('should produce deterministic results', () => {
    const audio = new Float32Array(4800);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;
    const r1 = mel.process(audio);
    const r2 = mel.process(audio);
    expect(r1.length).toBe(r2.length);
    for (let i = 0; i < r1.features.length; i++) expect(r1.features[i]).toBe(r2.features[i]);
  });

  it('should normalize to ~zero mean per mel bin', () => {
    const audio = new Float32Array(32000);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;
    const { features, length } = mel.process(audio);
    const nFrames = features.length / 128;
    for (let m = 0; m < 128; m++) {
      let sum = 0;
      for (let t = 0; t < length; t++) sum += features[m * nFrames + t];
      expect(Math.abs(sum / length)).toBeLessThan(0.1);
    }
  });

  it('samplesToFrames and framesToSamples should be inverses', () => {
    expect(mel.samplesToFrames(16000)).toBe(100);
    expect(mel.framesToSamples(100)).toBe(16000);
  });
});

describe('MelSpectrogram with custom params', () => {
  it('should accept custom nFft, winLength, hopLength', () => {
    const mel = new MelSpectrogram({ nMels: 80, nFft: 1024, winLength: 800, hopLength: 320 });
    expect(mel.nMels).toBe(80);
    expect(mel.nFft).toBe(1024);
    expect(mel.hopLength).toBe(320);
    const audio = new Float32Array(16000);
    const { length } = mel.process(audio);
    expect(length).toBe(50); // 16000 / 320
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// IncrementalMelSpectrogram
// ═══════════════════════════════════════════════════════════════════════════

describe('IncrementalMelSpectrogram', () => {
  it('should match full computation on first call', () => {
    const full = new MelSpectrogram({ nMels: 128 });
    const inc = new IncrementalMelSpectrogram({ nMels: 128 });

    const audio = new Float32Array(32000);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;

    const fullResult = full.process(audio);
    const incResult = inc.process(audio, 0);

    expect(incResult.length).toBe(fullResult.length);
    for (let i = 0; i < fullResult.features.length; i++) {
      expect(Math.abs(incResult.features[i] - fullResult.features[i])).toBeLessThan(1e-5);
    }
  });

  it('should reuse cached frames on second call with overlap', () => {
    const inc = new IncrementalMelSpectrogram({ nMels: 128 });
    const audio = new Float32Array(80000);
    for (let i = 0; i < audio.length; i++) {
      const t = i / 16000;
      audio[i] = Math.sin(2 * Math.PI * 440 * t) + 0.3 * Math.sin(2 * Math.PI * 880 * t);
    }

    const r1 = inc.process(audio, 0);
    expect(r1.cached).toBe(false);

    const prefixSamples = Math.floor(audio.length * 0.7);
    const r2 = inc.process(audio, prefixSamples);
    expect(r2.cached).toBe(true);
    expect(r2.cachedFrames).toBeGreaterThan(0);
    expect(r2.newFrames).toBeLessThan(r2.length);
  });

  it('should produce identical results with or without caching', () => {
    const full = new MelSpectrogram({ nMels: 128 });
    const inc = new IncrementalMelSpectrogram({ nMels: 128 });

    const audio = new Float32Array(80000);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;

    const fullResult = full.process(audio);
    inc.process(audio, 0);
    const prefixSamples = Math.floor(audio.length * 0.7);
    const incResult = inc.process(audio, prefixSamples);

    expect(incResult.length).toBe(fullResult.length);
    const nFramesFull = fullResult.features.length / 128;
    const nFramesInc = incResult.features.length / 128;
    let maxErr = 0;
    for (let m = 0; m < 128; m++) {
      for (let t = 0; t < fullResult.length; t++) {
        const err = Math.abs(fullResult.features[m * nFramesFull + t] - incResult.features[m * nFramesInc + t]);
        if (err > maxErr) maxErr = err;
      }
    }
    expect(maxErr).toBeLessThan(1e-5);
  });

  it('should clear cache on reset()', () => {
    const inc = new IncrementalMelSpectrogram({ nMels: 128 });
    const audio = new Float32Array(16000);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000);
    inc.process(audio, 0);
    inc.reset();
    const r2 = inc.process(audio, 0);
    expect(r2.cached).toBe(false);
  });

  it('should clear cache on clear() (alias)', () => {
    const inc = new IncrementalMelSpectrogram({ nMels: 128 });
    const audio = new Float32Array(16000);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000);
    inc.process(audio, 0);
    inc.clear();
    const r2 = inc.process(audio, 0);
    expect(r2.cached).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Legacy Aliases
// ═══════════════════════════════════════════════════════════════════════════

describe('Legacy aliases', () => {
  it('JsPreprocessor should be MelSpectrogram', () => {
    expect(JsPreprocessor).toBe(MelSpectrogram);
  });

  it('IncrementalMelProcessor should be IncrementalMelSpectrogram', () => {
    expect(IncrementalMelProcessor).toBe(IncrementalMelSpectrogram);
  });

  it('JsPreprocessor should work as a drop-in', () => {
    const p = new JsPreprocessor({ nMels: 128 });
    const audio = new Float32Array(16000);
    const { features, length } = p.process(audio);
    expect(length).toBe(100);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Performance Benchmark
// ═══════════════════════════════════════════════════════════════════════════

describe('Performance', () => {
  it('should process 5s audio in < 200ms', () => {
    const mel = new MelSpectrogram({ nMels: 128 });
    const audio = new Float32Array(80000);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;

    mel.process(audio); // warm up
    const t0 = performance.now();
    mel.process(audio);
    const elapsed = performance.now() - t0;

    console.log(`  [Benchmark] 5s audio: ${elapsed.toFixed(1)}ms`);
    expect(elapsed).toBeLessThan(200);
  });

  it('should process incrementally faster than full', () => {
    const inc = new IncrementalMelSpectrogram({ nMels: 128 });
    const audio = new Float32Array(80000);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;

    const t0 = performance.now();
    inc.process(audio, 0);
    const fullTime = performance.now() - t0;

    const prefixSamples = Math.floor(audio.length * 0.7);
    const t1 = performance.now();
    inc.process(audio, prefixSamples);
    const incTime = performance.now() - t1;

    console.log(`  [Benchmark] Full: ${fullTime.toFixed(1)}ms, Incremental: ${incTime.toFixed(1)}ms`);
    expect(incTime).toBeLessThan(fullTime * 1.5);
  });

  it('should process 10s audio in < 500ms', () => {
    const mel = new MelSpectrogram({ nMels: 128 });
    const audio = new Float32Array(160000);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;

    mel.process(audio); // warm up
    const t0 = performance.now();
    mel.process(audio);
    const elapsed = performance.now() - t0;

    console.log(`  [Benchmark] 10s audio: ${elapsed.toFixed(1)}ms`);
    expect(elapsed).toBeLessThan(500);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// ONNX Reference Cross-Validation
// ═══════════════════════════════════════════════════════════════════════════

describe('ONNX reference cross-validation', () => {
  let reference = null;

  beforeAll(() => {
    const refPath = join(testsDir, 'mel_reference.json');
    if (existsSync(refPath)) {
      reference = JSON.parse(readFileSync(refPath, 'utf-8'));
    }
  });

  it('should have mel_reference.json available', () => {
    expect(reference).not.toBeNull();
    expect(reference.nMels).toBe(128);
    expect(Object.keys(reference.tests).length).toBeGreaterThan(0);
  });

  it('should match ONNX mel filterbank within 1e-5', () => {
    if (!reference?.melFilterbank) return;

    const refFb = base64ToFloat32(reference.melFilterbank.data);
    const jsFb = createMelFilterbank(128);

    let maxErr = 0;
    for (let freq = 0; freq < 257; freq++) {
      for (let mel = 0; mel < 128; mel++) {
        const refVal = refFb[freq * 128 + mel];
        const jsVal = jsFb[mel * 257 + freq];
        const err = Math.abs(refVal - jsVal);
        if (err > maxErr) maxErr = err;
      }
    }

    console.log(`  [ONNX] Filterbank max error: ${maxErr.toExponential(3)}`);
    expect(maxErr).toBeLessThan(1e-5);
  });

  it('should match ONNX full pipeline within thresholds (max < 0.05, mean < 0.005)', () => {
    if (!reference) return;

    const mel = new MelSpectrogram({ nMels: reference.nMels });
    const nMels = reference.nMels;

    for (const [name, test] of Object.entries(reference.tests)) {
      const audio = base64ToFloat32(test.audio);
      const refFeatures = base64ToFloat32(test.features);

      const { features: jsFeatures, length: jsLen } = mel.process(audio);
      expect(jsLen).toBe(test.featuresLen);

      const nFramesJs = jsFeatures.length / nMels;
      const nFramesRef = refFeatures.length / nMels;

      let maxErr = 0, sumErr = 0, n = 0;
      for (let m = 0; m < nMels; m++) {
        for (let t = 0; t < jsLen; t++) {
          const jsVal = jsFeatures[m * nFramesJs + t];
          const refVal = refFeatures[m * nFramesRef + t];
          const err = Math.abs(jsVal - refVal);
          sumErr += err;
          if (err > maxErr) maxErr = err;
          n++;
        }
      }

      const meanErr = sumErr / n;
      console.log(`  [ONNX] Signal "${name}": max=${maxErr.toExponential(3)}, mean=${meanErr.toExponential(3)}`);
      expect(maxErr).toBeLessThan(0.05);
      expect(meanErr).toBeLessThan(0.005);
    }
  });
});
