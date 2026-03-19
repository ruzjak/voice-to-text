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
 * OUT { type: 'chunk_progress',  current, total, progress, partialText, statusLabel, processedSeconds, totalSeconds, timeRemaining }
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

const MODEL    = "onnx-community/whisper-large-v3-turbo";
const CHUNK_S  = 30;
const STRIDE_S = 5;
const SR       = 16_000;

/** 200 ms of silence prepended/appended to each chunk.
 *  Prevents RNNoise from clipping the first/last phonemes at boundaries. */
const VAD_PAD_S       = 0.2;
const VAD_PAD_SAMPLES = Math.round(VAD_PAD_S * SR); // 3 200 samples

/** How many tokens of previous chunk text to pass as a context prompt.
 *  Whisper's total token budget is 448; 128 leaves room for the new chunk. */
const PROMPT_TOKENS = 128;

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

// ── Chunking with VAD padding ─────────────────────────────────────────────────
//
// Each chunk gets VAD_PAD_SAMPLES of silence prepended and appended.
// This prevents the denoiser from clipping the first/last phonemes at the
// boundary, and gives Whisper a clean lead-in.

const SILENCE = new Float32Array(VAD_PAD_SAMPLES); // all-zero, reusable

function buildChunks(audio) {
  const chunkLen = CHUNK_S  * SR;               // 480 000 samples
  const step     = (CHUNK_S - STRIDE_S) * SR;   // 400 000 samples (25 s advance)
  const chunks   = [];
  let start = 0;
  while (start < audio.length) {
    const end  = Math.min(start + chunkLen, audio.length);
    const core = audio.slice(start, end);

    // Pad: [silence | core | silence]
    const padded = new Float32Array(SILENCE.length + core.length + SILENCE.length);
    padded.set(SILENCE, 0);
    padded.set(core,    SILENCE.length);
    padded.set(SILENCE, SILENCE.length + core.length);

    chunks.push({ slice: padded, startSec: start / SR, coreSamples: core.length });
    if (end >= audio.length) break;
    start += step;
  }
  console.log(
    `[worker] buildChunks — audio ${(audio.length / SR).toFixed(1)} s → ` +
    `${chunks.length} chunk(s) of ${CHUNK_S} s with ${STRIDE_S} s stride ` +
    `+ ${VAD_PAD_S * 1000} ms VAD padding each side`
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
    record(wallMs, audioSecs) {
      times.push(wallMs / audioSecs); // ms per audio-second
      if (times.length > ETA_WINDOW) times.shift();
    },
    estimate(remainingAudioSecs, progressFraction) {
      if (times.length === 0 || progressFraction < 0.05) return null;
      const avgMsPerAudioSec = times.reduce((a, b) => a + b, 0) / times.length;
      return (avgMsPerAudioSec * remainingAudioSecs) / 1000;
    },
  };
}

// ── Prompt helper ─────────────────────────────────────────────────────────────
//
// Pass the tail of the previous chunk's text as a context prompt so Whisper
// maintains vocabulary continuity across chunk boundaries (reduces repetition
// loops and hallucinated "..." runs on quiet lecture segments).

function buildPrompt(parts) {
  if (parts.length === 0) return undefined;
  // Join all prior parts, take the last PROMPT_TOKENS characters as a rough
  // token proxy (Czech averages ~4 chars/token with wordpiece).
  const joined = parts.join(" ");
  return joined.slice(-PROMPT_TOKENS * 4); // ~128 tokens
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
    const { slice, startSec, coreSamples } = chunks[i];
    const label = total === 1
      ? "Transcribing…"
      : `Transcribing chunk ${i + 1} of ${total}…`;

    console.log(
      `[worker] Inference chunk ${i + 1}/${total} START — ` +
      `offset ${startSec.toFixed(1)} s, ` +
      `core ${(coreSamples / SR).toFixed(1)} s + ${VAD_PAD_S * 1000} ms padding each side`
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

      // ── Anti-artifact thresholds ───────────────────────────────────────
      // logprob_threshold: allow the model to be more persistent with quiet
      //   speech — default is -1.0, raising to -0.6 keeps more weak tokens.
      // no_speech_threshold: 0.1 prevents Whisper from silencing quiet segments
      //   it would otherwise mark as non-speech (default 0.6 is too aggressive
      //   for lecture audio with background noise).
      // compression_ratio_threshold: 2.6 permits natural lecture repetition
      //   (formulas, enumerations) that the default 2.4 would flag as loops.
      logprob_threshold:           -0.6,
      no_speech_threshold:          0.1,
      compression_ratio_threshold:  2.6,

      // ── Cross-chunk context prompt ─────────────────────────────────────
      // Feed the tail of the previous chunk's text so the decoder maintains
      // vocabulary continuity across boundaries (Czech proper nouns, terms).
      ...(prompt !== undefined ? { prompt } : {}),

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
    eta.record(wallMs, coreSamples / SR);

    const text = (Array.isArray(output) ? output[0].text : output.text).trim();
    if (text) parts.push(text);

    const processedSeconds  = Math.min(startSec + coreSamples / SR, totalSeconds);
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
