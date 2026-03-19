/**
 * public/worker.js — Whisper transcription Web Worker
 *
 * Loaded as a module worker: new Worker('/worker.js', { type: 'module' })
 * Imports @huggingface/transformers v3 from jsDelivr CDN (bypasses webpack).
 * COEP is set to "credentialless" in next.config.ts to allow CDN imports.
 *
 * Message protocol
 * ────────────────
 * IN  { type: 'load' }
 * IN  { type: 'transcribe', audio: Float32Array }
 *
 * OUT { type: 'status',          status: 'loading' | 'ready' }
 * OUT { type: 'device',          device: 'webgpu' | 'wasm' }
 * OUT { type: 'model_progress',  file: string, progress: number }
 * OUT { type: 'chunk_progress',  current, total, progress, partialText,
 *                                statusLabel, processedSeconds, totalSeconds,
 *                                timeRemaining }
 * OUT { type: 'result',          text: string }
 * OUT { type: 'error',           message: string }
 */

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowLocalModels              = false;
env.useBrowserCache               = true;
env.backends.onnx.wasm.proxy      = true;
env.backends.onnx.wasm.numThreads = navigator.hardwareConcurrency || 4;

const MODEL    = "onnx-community/whisper-large-v3-turbo";
const CHUNK_S  = 30;
const STRIDE_S = 5;
const SR       = 16_000;

/** 200 ms of silence prepended/appended to each chunk (VAD padding). */
const VAD_PAD_S       = 0.2;
const VAD_PAD_SAMPLES = Math.round(VAD_PAD_S * SR); // 3 200 samples
const SILENCE         = new Float32Array(VAD_PAD_SAMPLES);

/** Approx chars to carry over as cross-chunk context (~64 tokens × 4 chars/tok). */
const PROMPT_CHARS = 256;

// ── WebGPU detection ─────────────────────────────────────────────────────────

async function detectWebGPU() {
  try {
    if (!("gpu" in navigator)) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

// ── Model singleton ───────────────────────────────────────────────────────────

let transcriber      = null;
let modelLoadPromise = null;

function getOrLoadModel() {
  if (transcriber)      return Promise.resolve(transcriber);
  if (modelLoadPromise) return modelLoadPromise;

  console.log("[worker] Model load START:", MODEL);
  self.postMessage({ type: "status", status: "loading" });

  modelLoadPromise = (async () => {
    const hasWebGPU = await detectWebGPU();
    const device    = hasWebGPU ? "webgpu" : "wasm";

    console.log(`[worker] Device selected: ${device}`);
    self.postMessage({ type: "device", device });

    const dtypeConfig = hasWebGPU
      ? { encoder_model: "fp16", decoder_model_merged: "q4" }
      : "q8";

    return pipeline("automatic-speech-recognition", MODEL, {
      device,
      dtype:     dtypeConfig,
      quantized: !hasWebGPU,
      progress_callback: (p) => {
        if (p.status === "progress") {
          const pct = Math.round(p.progress ?? 0);
          console.log(`[worker] model_progress — ${p.file} ${pct}%`);
          self.postMessage({ type: "model_progress", file: p.file ?? "", progress: pct });
        }
      },
    });
  })().then((pipe) => {
    transcriber = pipe;
    console.log("[worker] Model load END — ready");
    self.postMessage({ type: "status", status: "ready" });
    return pipe;
  });

  return modelLoadPromise;
}

// ── Chunking ──────────────────────────────────────────────────────────────────

function buildChunks(audio) {
  const chunkLen = CHUNK_S  * SR;
  const step     = (CHUNK_S - STRIDE_S) * SR;
  const chunks   = [];
  let start = 0;
  while (start < audio.length) {
    const end  = Math.min(start + chunkLen, audio.length);
    const core = audio.slice(start, end);

    // VAD padding: 200 ms silence on each side prevents boundary clipping
    const slice = new Float32Array(SILENCE.length + core.length + SILENCE.length);
    slice.set(SILENCE, 0);
    slice.set(core,    SILENCE.length);
    slice.set(SILENCE, SILENCE.length + core.length);

    chunks.push({ slice, startSec: start / SR, coreSamples: core.length });
    if (end >= audio.length) break;
    start += step;
  }
  return chunks;
}

// ── ETA helpers ───────────────────────────────────────────────────────────────

const ETA_WINDOW = 3;

function makeEtaTracker() {
  const times = [];
  return {
    record(wallMs, audioSecs) {
      times.push(wallMs / audioSecs);
      if (times.length > ETA_WINDOW) times.shift();
    },
    estimate(remainingAudioSecs, progressFraction) {
      if (times.length === 0 || progressFraction < 0.05) return null;
      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      return (avg * remainingAudioSecs) / 1000;
    },
  };
}

// ── Prompt helper ─────────────────────────────────────────────────────────────

function buildPrompt(parts) {
  if (parts.length === 0) return undefined;
  return parts.join(" ").slice(-PROMPT_CHARS);
}

// ── Transcription ─────────────────────────────────────────────────────────────

async function transcribeAudio(audio) {
  const pipe = await getOrLoadModel();

  const totalSeconds = audio.length / SR;
  const chunks       = buildChunks(audio);
  const total        = chunks.length;
  const parts        = [];
  const eta          = makeEtaTracker();

  console.group(
    `[worker] Inference START — ${total} chunk(s), total ${totalSeconds.toFixed(1)} s`
  );

  self.postMessage({
    type: "chunk_progress",
    current: 0, total, progress: 0,
    partialText:      "",
    statusLabel:      "Starting transcription…",
    processedSeconds: 0,
    totalSeconds,
    timeRemaining:    null,
  });

  for (let i = 0; i < total; i++) {
    const { slice, startSec, coreSamples } = chunks[i];
    const label = total === 1
      ? "Transcribing…"
      : `Transcribing chunk ${i + 1} of ${total}…`;

    console.log(
      `[worker] Chunk ${i + 1}/${total} START — ` +
      `offset ${startSec.toFixed(1)} s, ${(coreSamples / SR).toFixed(1)} s`
    );
    const t = performance.now();

    const prompt = buildPrompt(parts);

    let tokenCount = 0;
    const output = await pipe(slice, {
      task:              "transcribe",
      language:          "cs",
      chunk_length_s:    CHUNK_S,
      stride_length_s:   STRIDE_S,
      return_timestamps: false,

      condition_on_previous_text: true,
      repetition_penalty:         1.1,
      num_beams:                  1,

      ...(prompt !== undefined ? { prompt } : {}),

      callback_function: (beams) => {
        tokenCount++;
        if (tokenCount % 10 === 0) {
          const processedSeconds  = Math.min(startSec + CHUNK_S, totalSeconds);
          const progressFraction  = (i + tokenCount / 500) / total;
          const remainingAudioSec = Math.max(0, totalSeconds - processedSeconds);
          self.postMessage({
            type:  "chunk_progress",
            current: i + 1, total,
            progress:         Math.round(progressFraction * 100),
            partialText:      parts.join(" "),
            statusLabel:      `${label} (${tokenCount} tokens)`,
            processedSeconds,
            totalSeconds,
            timeRemaining:    eta.estimate(remainingAudioSec, progressFraction),
          });
        }
      },
    });

    const wallMs = performance.now() - t;
    eta.record(wallMs, coreSamples / SR);

    const text = (Array.isArray(output) ? output[0].text : output.text).trim();
    if (text) parts.push(text);

    const processedSeconds  = Math.min(startSec + coreSamples / SR, totalSeconds);
    const progressFraction  = (i + 1) / total;
    const remainingAudioSec = Math.max(0, totalSeconds - processedSeconds);

    console.log(
      `[worker] Chunk ${i + 1}/${total} END — ${wallMs.toFixed(0)} ms — ` +
      `"${text.slice(0, 60)}${text.length > 60 ? "…" : ""}"`
    );

    self.postMessage({
      type:  "chunk_progress",
      current: i + 1, total,
      progress:         Math.round(progressFraction * 100),
      partialText:      parts.join(" "),
      statusLabel:      total === 1 ? "Done." : `Chunk ${i + 1} of ${total} done.`,
      processedSeconds,
      totalSeconds,
      timeRemaining:    eta.estimate(remainingAudioSec, progressFraction),
    });
  }

  console.groupEnd();
  const finalText = parts.join(" ").trim();
  console.log(`[worker] Inference COMPLETE — "${finalText.slice(0, 120)}"`);

  self.postMessage({ type: "result", text: finalText });
}

// ── Message handler ───────────────────────────────────────────────────────────

self.addEventListener("message", async (e) => {
  try {
    if (e.data.type === "load") {
      await getOrLoadModel();
    } else if (e.data.type === "transcribe") {
      await transcribeAudio(e.data.audio);
    }
  } catch (err) {
    console.error("[worker] ERROR:", err);
    self.postMessage({ type: "error", message: err?.message ?? String(err) });
  }
});
