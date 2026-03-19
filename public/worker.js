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
 * IN  { type: 'transcribe', audio: Float32Array }   (transferable)
 *
 * OUT { type: 'status',          status: 'loading' | 'ready' }
 * OUT { type: 'device',          device: 'webgpu' | 'wasm' }
 * OUT { type: 'model_progress',  file: string, progress: number }
 * OUT { type: 'chunk_progress',  current, total, progress, partialText, statusLabel, processedSeconds, totalSeconds }
 * OUT { type: 'result',          text: string }
 * OUT { type: 'error',           message: string }
 */

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowLocalModels   = false;
env.useBrowserCache    = true;
// proxy=true offloads ONNX runtime into a sub-worker; this is a no-op when
// we are already inside a Worker, but harmless and future-safe for main-thread
// usage.
env.backends.onnx.wasm.proxy = true;

const MODEL   = "onnx-community/whisper-large-v3-turbo";
const CHUNK_S = 30;
const STRIDE_S = 5;
const SR      = 16_000;

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

// ── Model singleton with load-once queuing ────────────────────────────────────

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

    // dtype strategy:
    //   WebGPU — FP16 encoder + 4-bit merged decoder (fast GPU path)
    //   WASM   — INT8 across the board (safe, ~balanced memory/speed)
    const dtypeConfig = hasWebGPU
      ? { encoder_model: "fp16", decoder_model_merged: "q4" }
      : "q8";

    return pipeline("automatic-speech-recognition", MODEL, {
      device,
      dtype: dtypeConfig,
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
  const chunkLen = CHUNK_S  * SR;               // 480 000 samples
  const step     = (CHUNK_S - STRIDE_S) * SR;   // 400 000 samples  (25 s advance)
  const chunks   = [];
  let start = 0;
  while (start < audio.length) {
    const end = Math.min(start + chunkLen, audio.length);
    chunks.push({ slice: audio.slice(start, end), startSec: start / SR });
    if (end >= audio.length) break;
    start += step;
  }
  console.log(
    `[worker] buildChunks — audio ${(audio.length / SR).toFixed(1)} s → ` +
    `${chunks.length} chunk(s) of ${CHUNK_S} s with ${STRIDE_S} s stride`
  );
  return chunks;
}

// ── ETA helpers ───────────────────────────────────────────────────────────────
//
// We keep a sliding window of the last WINDOW real-wall seconds-per-audio-second
// ratios (one entry per completed chunk).  The average of those gives a stable
// estimate that ignores cold-start variance on the first chunk.
//
// timeRemaining = null  → still in the "Calculating…" phase (< 5 % processed
//                         OR no completed chunks yet).

const ETA_WINDOW = 3; // chunks

function makeEtaTracker() {
  const times = []; // wall-ms taken by each completed chunk
  return {
    /**
     * Record that one chunk of `audioSecs` seconds took `wallMs` ms.
     * @param {number} wallMs
     * @param {number} audioSecs — duration of the audio slice just processed
     */
    record(wallMs, audioSecs) {
      times.push(wallMs / audioSecs); // ms per audio-second
      if (times.length > ETA_WINDOW) times.shift();
    },
    /**
     * @param {number} remainingAudioSecs
     * @param {number} progressFraction  — 0–1, total fraction done
     * @returns {number|null}  remaining wall seconds, or null when too early
     */
    estimate(remainingAudioSecs, progressFraction) {
      if (times.length === 0 || progressFraction < 0.05) return null;
      const avgMsPerAudioSec = times.reduce((a, b) => a + b, 0) / times.length;
      return (avgMsPerAudioSec * remainingAudioSecs) / 1000;
    },
  };
}

// ── Transcription ─────────────────────────────────────────────────────────────

async function transcribeAudio(audio) {
  const pipe = await getOrLoadModel();

  const totalSeconds = audio.length / SR;
  const chunks       = buildChunks(audio);
  const total        = chunks.length;
  const parts        = [];
  const eta          = makeEtaTracker();

  console.group(`[worker] Inference START — ${total} chunk(s), total ${totalSeconds.toFixed(1)} s`);

  // Report 0 % before first chunk so the UI enters "transcribing" state
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
    const { slice, startSec } = chunks[i];
    const label = total === 1
      ? "Transcribing…"
      : `Transcribing chunk ${i + 1} of ${total}…`;

    console.log(
      `[worker] Inference chunk ${i + 1}/${total} START — ` +
      `offset ${startSec.toFixed(1)} s, length ${(slice.length / SR).toFixed(1)} s`
    );
    const t = performance.now();

    let tokenCount = 0;
    const output = await pipe(slice, {
      task:              "transcribe",
      language:          "cs",
      chunk_length_s:    CHUNK_S,
      stride_length_s:   STRIDE_S,
      return_timestamps: false,
      callback_function: () => {
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
    eta.record(wallMs, slice.length / SR);

    const text = (Array.isArray(output) ? output[0].text : output.text).trim();
    if (text) parts.push(text);

    const processedSeconds  = Math.min(startSec + slice.length / SR, totalSeconds);
    const progressFraction  = (i + 1) / total;
    const remainingAudioSec = Math.max(0, totalSeconds - processedSeconds);
    console.log(
      `[worker] Inference chunk ${i + 1}/${total} END — ` +
      `${wallMs.toFixed(0)} ms — ` +
      `processed up to ${processedSeconds.toFixed(1)} s — ` +
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
  console.log(`[worker] Inference COMPLETE — final text: "${finalText.slice(0, 120)}"`);
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
