import { describe, it, expect } from "vitest";
import {
  PARAKEET_MEL_VARIANTS,
  createParakeetMelProcessor,
  ParakeetIncrementalMelProcessor,
} from "../src/index.js";

function makeAudio(samples) {
  const audio = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    const t = i / 16000;
    audio[i] =
      0.4 * Math.sin(2 * Math.PI * 440 * t) +
      0.3 * Math.sin(2 * Math.PI * 660 * t) +
      0.1 * Math.sin(2 * Math.PI * 880 * t);
  }
  return audio;
}

function maxAbsDiffByValidFrames(a, b, nMels) {
  const nFramesA = a.features.length / nMels;
  const nFramesB = b.features.length / nMels;
  const valid = Math.min(a.length, b.length);
  let max = 0;

  for (let m = 0; m < nMels; m++) {
    const baseA = m * nFramesA;
    const baseB = m * nFramesB;
    for (let t = 0; t < valid; t++) {
      const d = Math.abs(a.features[baseA + t] - b.features[baseB + t]);
      if (d > max) max = d;
    }
  }
  return max;
}

describe("Parakeet mel variants", () => {
  it("should expose all expected variant keys", () => {
    expect(PARAKEET_MEL_VARIANTS).toEqual(["current", "pr74", "pr75", "pr84"]);
  });

  it("each variant should run and match baseline numerically", () => {
    const audio = makeAudio(32000); // 2s
    const baseline = createParakeetMelProcessor({ variant: "current", nMels: 128 });
    const baselineOut = baseline.process(audio);

    for (const variant of PARAKEET_MEL_VARIANTS) {
      const proc = createParakeetMelProcessor({ variant, nMels: 128 });
      const out = proc.process(audio);
      expect(out.length).toBe(baselineOut.length);
      expect(out.features.length).toBe(baselineOut.features.length);
      const maxDiff = maxAbsDiffByValidFrames(out, baselineOut, 128);
      expect(maxDiff).toBeLessThan(1e-6);
    }
  });

  it("variant aliases should resolve", () => {
    const a = createParakeetMelProcessor({ variant: "pr-74" });
    const b = createParakeetMelProcessor({ variant: "candidate75" });
    const c = createParakeetMelProcessor({ variant: "realfft84" });
    expect(a.variant).toBe("pr74");
    expect(b.variant).toBe("pr75");
    expect(c.variant).toBe("pr84");
  });

  it("incremental processor should cache and remain numerically stable", () => {
    const audio = makeAudio(80000); // 5s
    const full = createParakeetMelProcessor({ variant: "pr75", nMels: 128 });
    const inc = new ParakeetIncrementalMelProcessor({ variant: "pr75", nMels: 128 });

    const fullOut = full.process(audio);
    const first = inc.process(audio, 0);
    expect(first.cached).toBe(false);
    expect(first.length).toBe(fullOut.length);

    const second = inc.process(audio, Math.floor(audio.length * 0.7));
    expect(second.cached).toBe(true);
    expect(second.cachedFrames).toBeGreaterThan(0);
    expect(second.newFrames).toBeLessThan(second.length);

    const maxDiff = maxAbsDiffByValidFrames(second, fullOut, 128);
    expect(maxDiff).toBeLessThan(1e-5);
  });
});

