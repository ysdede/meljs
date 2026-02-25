#!/usr/bin/env node
/**
 * Benchmark and accuracy comparison for parakeet mel variants in meljs.
 *
 * Usage:
 *   node tests/benchmark_variants.mjs
 *   node tests/benchmark_variants.mjs --audio "N:\\path\\file.wav" --splits "1,2,6,10,16,24,32,38,50,80,whole"
 *   node tests/benchmark_variants.mjs --splits "7,8,9,10,11,12,13,14" --runs 8
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import {
  PARAKEET_MEL_VARIANTS,
  createParakeetMelProcessor,
} from "../src/index.js";

const DEFAULT_AUDIO =
  "N:\\github\\ysdede\\scribe-ds\\data\\PARROT_v1.0\\07_audio\\labels\\full_reports\\b9c6e6e1a552d7ef7fe68b775f78a9f3dbffee2930f9cf589c70d3030cd1b414.wav";
const DEFAULT_SPLITS = "1,2,6,10,16,24,32,38,50,80,whole";

function parseArgs(argv) {
  const out = {
    audio: DEFAULT_AUDIO,
    splits: DEFAULT_SPLITS,
    runs: 8,
    warmup: 3,
    nMels: 128,
    out: "",
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--audio" && argv[i + 1]) out.audio = argv[++i];
    else if (a === "--splits" && argv[i + 1]) out.splits = argv[++i];
    else if (a === "--runs" && argv[i + 1]) out.runs = Number(argv[++i]);
    else if (a === "--warmup" && argv[i + 1]) out.warmup = Number(argv[++i]);
    else if (a === "--nMels" && argv[i + 1]) out.nMels = Number(argv[++i]);
    else if (a === "--out" && argv[i + 1]) out.out = argv[++i];
  }
  return out;
}

function readAscii(view, offset, len) {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(offset + i));
  return s;
}

function decodeWavToMonoFloat32(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (readAscii(view, 0, 4) !== "RIFF" || readAscii(view, 8, 4) !== "WAVE") {
    throw new Error("Unsupported WAV header");
  }

  let fmt = null;
  let dataOffset = -1;
  let dataSize = -1;
  let pos = 12;

  while (pos + 8 <= view.byteLength) {
    const id = readAscii(view, pos, 4);
    const size = view.getUint32(pos + 4, true);
    const chunkData = pos + 8;
    if (id === "fmt ") {
      fmt = {
        audioFormat: view.getUint16(chunkData, true),
        numChannels: view.getUint16(chunkData + 2, true),
        sampleRate: view.getUint32(chunkData + 4, true),
        blockAlign: view.getUint16(chunkData + 12, true),
        bitsPerSample: view.getUint16(chunkData + 14, true),
      };
    } else if (id === "data") {
      dataOffset = chunkData;
      dataSize = size;
      break;
    }
    pos = chunkData + size + (size & 1);
  }

  if (!fmt || dataOffset < 0 || dataSize <= 0) {
    throw new Error("WAV fmt/data chunk missing");
  }

  const bytesPerSample = fmt.bitsPerSample >> 3;
  const frames = Math.floor(dataSize / fmt.blockAlign);
  const out = new Float32Array(frames);

  const readSample = (byteOffset) => {
    if (fmt.audioFormat === 1) {
      if (fmt.bitsPerSample === 16) return view.getInt16(byteOffset, true) / 32768;
      if (fmt.bitsPerSample === 24) {
        const b0 = view.getUint8(byteOffset);
        const b1 = view.getUint8(byteOffset + 1);
        const b2 = view.getUint8(byteOffset + 2);
        let v = b0 | (b1 << 8) | (b2 << 16);
        if (v & 0x800000) v |= 0xff000000;
        return v / 8388608;
      }
      if (fmt.bitsPerSample === 32) return view.getInt32(byteOffset, true) / 2147483648;
      if (fmt.bitsPerSample === 8) return (view.getUint8(byteOffset) - 128) / 128;
    }
    if (fmt.audioFormat === 3 && fmt.bitsPerSample === 32) return view.getFloat32(byteOffset, true);
    throw new Error(`Unsupported WAV format=${fmt.audioFormat} bits=${fmt.bitsPerSample}`);
  };

  for (let i = 0; i < frames; i++) {
    const frameOffset = dataOffset + i * fmt.blockAlign;
    let acc = 0;
    for (let c = 0; c < fmt.numChannels; c++) {
      acc += readSample(frameOffset + c * bytesPerSample);
    }
    out[i] = acc / fmt.numChannels;
  }

  return {
    audio: out,
    sampleRate: fmt.sampleRate,
    numChannels: fmt.numChannels,
    bitsPerSample: fmt.bitsPerSample,
    audioFormat: fmt.audioFormat,
    durationSec: out.length / fmt.sampleRate,
  };
}

function makeSyntheticAudio(seconds = 120, sampleRate = 16000) {
  const n = Math.floor(seconds * sampleRate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    out[i] =
      0.4 * Math.sin(2 * Math.PI * 220 * t) +
      0.2 * Math.sin(2 * Math.PI * 440 * t) +
      0.1 * Math.sin(2 * Math.PI * 880 * t);
  }
  return { audio: out, sampleRate, durationSec: seconds, synthetic: true };
}

function parseSplits(raw, sampleRate, totalSamples) {
  const parts = raw.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
  const seen = new Set();
  const splits = [];
  for (const p of parts) {
    let samples;
    let requestedSec = null;
    if (p === "whole") {
      samples = totalSamples;
    } else {
      const sec = Number(p);
      if (!Number.isFinite(sec) || sec <= 0) continue;
      requestedSec = sec;
      samples = Math.min(totalSamples, Math.floor(sec * sampleRate));
    }
    if (samples <= 0 || seen.has(samples)) continue;
    seen.add(samples);
    splits.push({
      label: p,
      requestedSec,
      samples,
      actualSec: samples / sampleRate,
      truncated: requestedSec != null && samples < Math.floor(requestedSec * sampleRate),
    });
  }
  return splits;
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const x = (sorted.length - 1) * p;
  const i0 = Math.floor(x);
  const i1 = Math.min(sorted.length - 1, i0 + 1);
  const t = x - i0;
  return sorted[i0] * (1 - t) + sorted[i1] * t;
}

function summarize(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const mean = times.reduce((a, x) => a + x, 0) / times.length;
  const variance = times.reduce((a, x) => a + (x - mean) * (x - mean), 0) / times.length;
  return {
    runs: times.length,
    minMs: sorted[0],
    p50Ms: percentile(sorted, 0.5),
    p90Ms: percentile(sorted, 0.9),
    maxMs: sorted[sorted.length - 1],
    meanMs: mean,
    stdMs: Math.sqrt(variance),
    allMs: times,
  };
}

function compareFeatures(out, baseline, nMels) {
  const frameStrideA = Math.floor(out.features.length / nMels);
  const frameStrideB = Math.floor(baseline.features.length / nMels);
  const validFrames = Math.min(out.length, baseline.length);

  let maxAbsDiff = 0;
  let sumAbs = 0;
  let sumSq = 0;
  let count = 0;

  for (let m = 0; m < nMels; m++) {
    const baseA = m * frameStrideA;
    const baseB = m * frameStrideB;
    for (let t = 0; t < validFrames; t++) {
      const d = out.features[baseA + t] - baseline.features[baseB + t];
      const ad = Math.abs(d);
      if (ad > maxAbsDiff) maxAbsDiff = ad;
      sumAbs += ad;
      sumSq += d * d;
      count++;
    }
  }

  return {
    comparedFrames: validFrames,
    frameLengthA: out.length,
    frameLengthB: baseline.length,
    maxAbsDiff,
    meanAbsDiff: count ? sumAbs / count : 0,
    rmse: count ? Math.sqrt(sumSq / count) : 0,
    count,
  };
}

function formatMs(v) {
  return v.toFixed(3).padStart(8);
}

function formatExp(v) {
  return v.toExponential(3).padStart(12);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const audioPath = resolve(args.audio);

  let source;
  if (existsSync(audioPath)) {
    source = decodeWavToMonoFloat32(readFileSync(audioPath));
    if (source.sampleRate !== 16000) {
      throw new Error(`Expected 16k audio for parakeet variants, got ${source.sampleRate}`);
    }
  } else {
    source = makeSyntheticAudio(120, 16000);
  }

  const splits = parseSplits(args.splits, 16000, source.audio.length);
  if (!splits.length) throw new Error("No valid splits to run");

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = args.out
    ? resolve(args.out)
    : resolve("tmp", `meljs-variant-benchmark-${timestamp}.json`);

  mkdirSync(dirname(outPath), { recursive: true });

  console.log("meljs variant benchmark");
  console.log(`  source: ${source.synthetic ? "synthetic" : audioPath}`);
  console.log(`  duration: ${source.durationSec.toFixed(3)}s, samples=${source.audio.length}`);
  console.log(`  splits: ${splits.map((s) => `${s.label}(${s.actualSec.toFixed(2)}s)`).join(", ")}`);
  console.log(`  variants: ${PARAKEET_MEL_VARIANTS.join(", ")}`);
  console.log(`  warmup=${args.warmup}, runs=${args.runs}, nMels=${args.nMels}`);

  const result = {
    meta: {
      generatedAt: new Date().toISOString(),
      source: source.synthetic ? { synthetic: true } : {
        audioPath,
        sampleRate: source.sampleRate,
        durationSec: source.durationSec,
        numChannels: source.numChannels,
        bitsPerSample: source.bitsPerSample,
        audioFormat: source.audioFormat,
      },
      splits,
      runs: args.runs,
      warmup: args.warmup,
      nMels: args.nMels,
      variants: PARAKEET_MEL_VARIANTS,
    },
    bySplit: [],
  };

  for (const split of splits) {
    const segment = source.audio.subarray(0, split.samples);
    const splitRec = {
      split,
      variants: {},
      accuracyVsCurrent: {},
    };
    const outputs = new Map();

    for (const variant of PARAKEET_MEL_VARIANTS) {
      const proc = createParakeetMelProcessor({ variant, nMels: args.nMels });
      for (let i = 0; i < args.warmup; i++) proc.process(segment);

      const times = [];
      let out = null;
      for (let r = 0; r < args.runs; r++) {
        const t0 = performance.now();
        out = proc.process(segment);
        times.push(performance.now() - t0);
      }

      const timing = summarize(times);
      outputs.set(variant, out);
      splitRec.variants[variant] = {
        timing: {
          ...timing,
          rtfxP50: split.actualSec / (timing.p50Ms / 1000),
        },
        output: {
          validFrames: out.length,
          featuresLength: out.features.length,
        },
      };
    }

    const baseline = outputs.get("current");
    const baselineP50 = splitRec.variants.current.timing.p50Ms;
    for (const variant of PARAKEET_MEL_VARIANTS) {
      const out = outputs.get(variant);
      splitRec.accuracyVsCurrent[variant] = compareFeatures(out, baseline, args.nMels);
      splitRec.variants[variant].timing.speedupVsCurrentP50 =
        baselineP50 / splitRec.variants[variant].timing.p50Ms;
    }

    result.bySplit.push(splitRec);
  }

  writeFileSync(outPath, JSON.stringify(result, null, 2));
  const latestPath = resolve("tmp", "meljs-variant-benchmark-latest.json");
  writeFileSync(latestPath, JSON.stringify(result, null, 2));
  const manifestPath = resolve("tmp", "meljs-variant-benchmark-manifest.json");
  let manifest = { reports: [] };
  if (existsSync(manifestPath)) {
    try {
      manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
      if (!manifest || !Array.isArray(manifest.reports)) {
        manifest = { reports: [] };
      }
    } catch {
      manifest = { reports: [] };
    }
  }

  const reportName = basename(outPath);
  const existingIdx = manifest.reports.findIndex((r) => r.path === reportName);
  const entry = {
    path: reportName,
    generatedAt: result.meta.generatedAt,
    splits: result.meta.splits.map((s) => s.label),
  };
  if (existingIdx >= 0) {
    manifest.reports[existingIdx] = entry;
  } else {
    manifest.reports.push(entry);
  }
  manifest.reports.sort((a, b) => String(a.generatedAt).localeCompare(String(b.generatedAt)));
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

  console.log("");
  for (const splitRec of result.bySplit) {
    const split = splitRec.split;
    console.log(`Split ${split.label} (${split.actualSec.toFixed(2)}s, ${split.samples} samples)`);
    console.log("  Variant          p50(ms)    p90(ms)       RTFx    speedup        maxAbsDiff   meanAbsDiff         rmse");
    for (const variant of PARAKEET_MEL_VARIANTS) {
      const t = splitRec.variants[variant].timing;
      const a = splitRec.accuracyVsCurrent[variant];
      console.log(
        `  ${variant.padEnd(13)} ${formatMs(t.p50Ms)} ${formatMs(t.p90Ms)} ${t.rtfxP50.toFixed(3).padStart(10)} ${t.speedupVsCurrentP50
          .toFixed(3)
          .padStart(10)} ${formatExp(a.maxAbsDiff)} ${formatExp(a.meanAbsDiff)} ${formatExp(a.rmse)}`
      );
    }
    console.log("");
  }

  console.log(`Report written: ${outPath}`);
  console.log(`Latest report: ${latestPath}`);
  console.log(`Manifest: ${manifestPath}`);
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
