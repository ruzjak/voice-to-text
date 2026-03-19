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
 * IN  { type: 'transcribe', audio: Float32Array, config?: InferenceConfig }
 *
 * InferenceConfig (all optional, defaults to "full quality" mode):
 *   debugMode        — log chunk-level diagnostics to the console
 *   enablePrompting  — pass prior-chunk text as context prompt (default: true)
 *   enableVadPadding — prepend/append 200ms silence to each chunk (default: true)
 *   enableThresholds — apply anti-artifact Whisper thresholds (default: true)
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

env.allowLocalModels   = false;
env.useBrowserCache    = true;
env.backends.onnx.wasm.proxy = true;

const MODEL    = "onnx-community/whisper-large-v3-turbo";
const CHUNK_S  = 30;
const STRIDE_S = 5;
const SR       = 16_000;

/** 200 ms of silence prepended/appended to each chunk (VAD padding). */
const VAD_PAD_S       = 0.2;
const VAD_PAD_SAMPLES = Math.round(VAD_PAD_S * SR); // 3 200 samples
const SILENCE         = new Float32Array(VAD_PAD_SAMPLES);

/** Approx chars to carry over as cross-chunk context (~128 tokens × 4 chars/tok). */
const PROMPT_CHARS = 512;

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

function buildChunks(audio, withPadding) {
  const chunkLen = CHUNK_S  * SR;
  const step     = (CHUNK_S - STRIDE_S) * SR;
  const chunks   = [];
  let start = 0;
  while (start < audio.length) {
    const end  = Math.min(start + chunkLen, audio.length);
    const core = audio.slice(start, end);

    let slice;
    if (withPadding) {
      slice = new Float32Array(SILENCE.length + core.length + SILENCE.length);
      slice.set(SILENCE, 0);
      slice.set(core,    SILENCE.length);
      slice.set(SILENCE, SILENCE.length + core.length);
    } else {
      slice = core;
    }

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

// ── Debug helpers ─────────────────────────────────────────────────────────────

/** RMS level of a Float32Array — useful for detecting silent chunks. */
function computeRMS(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / (samples.length || 1));
}

/** Peak absolute value of a Float32Array. */
function computePeak(samples) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
  }
  return peak;
}

// ── Prompt helper ─────────────────────────────────────────────────────────────

function buildPrompt(parts) {
  if (parts.length === 0) return undefined;
  return parts.join(" ").slice(-PROMPT_CHARS);
}

// ── Transcription ─────────────────────────────────────────────────────────────

async function transcribeAudio(audio, config) {
  const {
    debugMode        = false,
    enablePrompting  = true,
    enableVadPadding = true,
    enableThresholds = true,
  } = config ?? {};

  const pipe = await getOrLoadModel();

  const totalSeconds = audio.length / SR;
  const chunks       = buildChunks(audio, enableVadPadding);
  const total        = chunks.length;
  const parts        = [];
  const eta          = makeEtaTracker();

  const modeLabel = debugMode
    ? `[DEBUG raw — prompting:${enablePrompting} padding:${enableVadPadding} thresholds:${enableThresholds}]`
    : "[normal]";

  console.group(
    `[worker] Inference START ${modeLabel} — ` +
    `${total} chunk(s), total ${totalSeconds.toFixed(1)} s`
  );

  if (debugMode) {
    const audioRMS  = computeRMS(audio);
    const audioPeak = computePeak(audio);
    console.log(
      `[debug] Full audio — RMS: ${audioRMS.toFixed(5)}, ` +
      `Peak: ${audioPeak.toFixed(5)}, ` +
      `Duration: ${totalSeconds.toFixed(2)} s, ` +
      `Samples: ${audio.length.toLocaleString()}`
    );
    console.log(
      `[debug] Config — enablePrompting: ${enablePrompting}, ` +
      `enableVadPadding: ${enableVadPadding}, ` +
      `enableThresholds: ${enableThresholds}`
    );
    console.log(
      debugMode
        ? "[debug] ⚠️  Using STOCK Whisper parameters (no anti-artifact thresholds)"
        : "[debug] Using tuned anti-artifact thresholds"
    );
  }

  self.postMessage({
    type: "chunk_progress",
    current: 0, total, progress: 0,
    partialText:      "",
    statusLabel:      debugMode ? "⚙️ Debug Mode — Starting…" : "Starting transcription…",
    processedSeconds: 0,
    totalSeconds,
    timeRemaining:    null,
  });

  for (let i = 0; i < total; i++) {
    const { slice, startSec, coreSamples } = chunks[i];
    const label = total === 1
      ? (debugMode ? "⚙️ Debug Transcribing…" : "Transcribing…")
      : (debugMode
          ? `⚙️ Debug chunk ${i + 1}/${total}…`
          : `Transcribing chunk ${i + 1} of ${total}…`);

    if (debugMode) {
      // Core slice (without padding) for accurate signal stats
      const core = audio.slice(
        Math.round(startSec * SR),
        Math.min(Math.round(startSec * SR) + coreSamples, audio.length)
      );
      const coreRMS  = computeRMS(core);
      const corePeak = computePeak(core);
      const silentPct = (() => {
        let silentCount = 0;
        for (let s = 0; s < core.length; s++) if (Math.abs(core[s]) < 0.001) silentCount++;
        return ((silentCount / core.length) * 100).toFixed(1);
      })();

      console.group(
        `[debug] ── Chunk ${i + 1}/${total} ──  ` +
        `offset ${startSec.toFixed(1)} s, core ${(coreSamples / SR).toFixed(1)} s`
      );
      console.log(
        `[debug] Signal: RMS=${coreRMS.toFixed(5)}, Peak=${corePeak.toFixed(5)}, ` +
        `~${silentPct}% near-silent samples (<0.001)`
      );
      console.log(
        `[debug] Padding: ${enableVadPadding ? `${VAD_PAD_S * 1000} ms each side` : "none"}`
      );
      const prompt = enablePrompting ? buildPrompt(parts) : undefined;
      console.log(
        `[debug] Prompt: ${prompt
          ? `"…${prompt.slice(-80)}" (${prompt.length} chars)`
          : "none"}`
      );
      if (coreRMS < 0.005) {
        console.warn(
          `[debug] ⚠️  VERY LOW SIGNAL on chunk ${i + 1} ` +
          `(RMS ${coreRMS.toFixed(5)}) — Whisper may output empty or hallucinated text`
        );
      }
    }

    console.log(
      `[worker] Chunk ${i + 1}/${total} START — ` +
      `offset ${startSec.toFixed(1)} s, ${(coreSamples / SR).toFixed(1)} s`
    );
    const t = performance.now();

    const prompt = enablePrompting ? buildPrompt(parts) : undefined;

    let tokenCount = 0;
    const output = await pipe(slice, {
      task:              "transcribe",
      language:          "cs",
      chunk_length_s:    CHUNK_S,
      stride_length_s:   STRIDE_S,
      return_timestamps: false,

      // Anti-artifact thresholds — skipped in debug/raw mode
      ...(enableThresholds ? {
        logprob_threshold:          -0.6,
        no_speech_threshold:         0.1,
        compression_ratio_threshold: 2.6,
      } : {}),

      // Cross-chunk context prompt — skipped when disabled
      ...(prompt !== undefined ? { prompt } : {}),

      callback_function: (beams) => {
        tokenCount++;

        // Debug: log beam scores every 50 tokens to surface logprob signals
        if (debugMode && tokenCount % 50 === 0 && beams?.length) {
          const beam = beams[0];
          console.log(
            `[debug] Token ${tokenCount} — ` +
            `score: ${beam.score?.toFixed(4) ?? "n/a"}, ` +
            `output_token_ids.length: ${beam.output_token_ids?.length ?? "n/a"}`
          );
        }

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

    if (debugMode) {
      const verdict = text
        ? `✅ "${text.slice(0, 100)}${text.length > 100 ? "…" : ""}"`
        : "⚠️  EMPTY — no speech detected by model";
      console.log(
        `[debug] Chunk ${i + 1} result: ${verdict}\n` +
        `        tokens=${tokenCount}, wall=${wallMs.toFixed(0)} ms, ` +
        `tokens/s=${(tokenCount / (wallMs / 1000)).toFixed(1)}`
      );
      console.groupEnd();
    }

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

  if (debugMode) {
    console.log("─".repeat(60));
    console.log(`[debug] ✅ FINAL RESULT (${finalText.length} chars, ${total} chunks)`);
    console.log(`[debug] "${finalText.slice(0, 300)}${finalText.length > 300 ? "…" : ""}"`);
    console.log("─".repeat(60));
  } else {
    console.log(`[worker] Inference COMPLETE — "${finalText.slice(0, 120)}"`);
  }

  self.postMessage({ type: "result", text: finalText });
}

// ── Message handler ───────────────────────────────────────────────────────────

self.addEventListener("message", async (e) => {
  try {
    if (e.data.type === "load") {
      await getOrLoadModel();
    } else if (e.data.type === "transcribe") {
      await transcribeAudio(e.data.audio, e.data.config);
    }
  } catch (err) {
    console.error("[worker] ERROR:", err);
    self.postMessage({ type: "error", message: err?.message ?? String(err) });
  }
});
