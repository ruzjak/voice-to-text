/**
 * public/denoise-worker.js — RNNoise offline denoising Web Worker
 *
 * Loaded as a module worker: new Worker('/denoise-worker.js', { type: 'module' })
 * The rnnoise-sync.js is served from /rnnoise/ (WASM binary embedded as base64).
 *
 * ── Segmented memory-safe pipeline ───────────────────────────────────────────
 * The main thread splits long audio into 1-minute segments and sends them one
 * at a time.  Each message carries one segment; the worker processes it and
 * immediately transfers the result back so the WASM heap can be released
 * before the next segment arrives.
 *
 * Within each segment the per-frame approach is used:
 *   One RNNoise frame = 480 samples @ 44 100 Hz = 10 ms.
 *   At 16 000 Hz that equals:  480 × (16000 / 44100) = 173.87 ≈ 174 samples.
 *   For each frame we:
 *     1. Read 174 samples from the segment (zero-copy subarray view).
 *     2. Upsample 174 → 480 into a pre-allocated reusable buffer.
 *     3. Run RNNoise in-place on the 480-sample buffer.
 *     4. Downsample 480 → 174 back into the segment (in-place).
 *   Peak extra memory per segment: one Float32Array(480) = 1 920 bytes.
 *
 * Message protocol
 * ────────────────
 * IN  { type: 'denoise_segment', segment: Float32Array }   (transferable, 16 kHz mono)
 *
 * OUT { type: 'segment_progress', progress: number }       (0–100 within this segment)
 * OUT { type: 'segment_result',   segment: Float32Array }  (transferable, processed)
 * OUT { type: 'error',            message: string }
 */

import createRNNWasmModuleSync from "/rnnoise/rnnoise-sync.js";

// ── Constants ─────────────────────────────────────────────────────────────────

/** RNNoise native sample rate (Hz). Input MUST be at this rate at the WASM boundary. */
const RNNOISE_SR         = 44100;
/** Target sample rate for Whisper (Hz). Segment arrives and leaves at this rate. */
const TARGET_SR          = 16000;
/** Exact samples per RNNoise frame at RNNOISE_SR. Non-negotiable — WASM crashes otherwise. */
const RNNOISE_FRAME_SIZE = 480;
/** Bytes occupied by one WASM frame buffer (float32). */
const RNNOISE_FRAME_BYTES = RNNOISE_FRAME_SIZE * 4;
/** int16 scale factor used by RNNoise internally. */
const SHIFT = 32768;

/**
 * How many TARGET_SR samples correspond to one RNNOISE_FRAME_SIZE frame.
 * = 480 × (16000 / 44100) = 173.87 → rounded to 174.
 */
const FRAME_SIZE_16K = Math.round(RNNOISE_FRAME_SIZE * TARGET_SR / RNNOISE_SR); // 174

/** Report progress every N frames (~10 s of audio per update within a segment). */
const PROGRESS_EVERY = 920;

// ── In-place linear resampler ─────────────────────────────────────────────────

function resampleInto(input, output) {
  const srcLen = input.length;
  const dstLen = output.length;
  const ratio  = srcLen / dstLen;
  const maxSrc = srcLen - 1;

  for (let i = 0; i < dstLen; i++) {
    const pos  = i * ratio;
    const lo   = pos | 0;
    const hi   = lo < maxSrc ? lo + 1 : maxSrc;
    const frac = pos - lo;
    output[i]  = input[lo] + frac * (input[hi] - input[lo]);
  }
}

// ── RnnoiseProcessor ──────────────────────────────────────────────────────────

class RnnoiseProcessor {
  constructor(wasmInterface) {
    this._wasmInterface = wasmInterface;
    this._wasmPcmInput  = wasmInterface._malloc(RNNOISE_FRAME_BYTES);
    if (!this._wasmPcmInput) {
      throw new Error(
        `RNNoise WASM _malloc(${RNNOISE_FRAME_BYTES}) returned 0 — WASM heap exhausted.`
      );
    }
    this._inputF32Idx = this._wasmPcmInput >> 2;
    this._context     = wasmInterface._rnnoise_create();
    this._destroyed   = false;
  }

  processAudioFrame(frame, shouldDenoise = false) {
    for (let i = 0; i < RNNOISE_FRAME_SIZE; i++) {
      this._wasmInterface.HEAPF32[this._inputF32Idx + i] = frame[i] * SHIFT;
    }
    const vad = this._wasmInterface._rnnoise_process_frame(
      this._context, this._wasmPcmInput, this._wasmPcmInput,
    );
    if (shouldDenoise) {
      for (let i = 0; i < RNNOISE_FRAME_SIZE; i++) {
        frame[i] = this._wasmInterface.HEAPF32[this._inputF32Idx + i] / SHIFT;
      }
    }
    return vad;
  }

  destroy() {
    if (this._destroyed) return;
    this._wasmInterface._free(this._wasmPcmInput);
    this._wasmInterface._rnnoise_destroy(this._context);
    this._destroyed = true;
  }
}

// ── Segment processing ────────────────────────────────────────────────────────
//
// Receives a single 1-min (or shorter) Float32Array at 16 kHz, denoises it
// frame-by-frame, and returns it modified in-place.
// A fresh WASM module is created and destroyed for each segment so the heap
// is released before the next segment is transferred in.

async function processSegment(pcm16k) {
  const totalSamples = pcm16k.length;
  const totalFrames  = Math.floor(totalSamples / FRAME_SIZE_16K);
  const segSecs      = (totalSamples / TARGET_SR).toFixed(1);

  console.log(
    `[denoise-worker] segment START — ${segSecs} s, ` +
    `${totalSamples.toLocaleString()} samples, ${totalFrames.toLocaleString()} frames`
  );

  // Heap-size advisory: segments should be ~3.8 MB; warn if unusually large
  const segMB = (totalSamples * 4) / (1024 * 1024);
  if (segMB > 50) {
    console.warn(
      `[denoise-worker] Segment is ${segMB.toFixed(1)} MB — ` +
      "consider reducing SEGMENT_SECS on the main thread."
    );
  }

  // Fresh WASM module per segment — heap is released when local vars go out of scope
  const wasmModule = createRNNWasmModuleSync();
  let processor;
  try {
    processor = new RnnoiseProcessor(wasmModule);
  } catch (err) {
    console.error(
      `[denoise-worker] RnnoiseProcessor init failed — ` +
      `segment ${totalSamples.toLocaleString()} samples (${(totalSamples * 4 / 1048576).toFixed(1)} MB):`,
      err
    );
    throw err;
  }

  const frame44k = new Float32Array(RNNOISE_FRAME_SIZE);

  // Report 0 % immediately so the UI bar advances on first frame
  self.postMessage({ type: "segment_progress", progress: 0 });

  const t0 = performance.now();

  for (let f = 0; f < totalFrames; f++) {
    const start   = f * FRAME_SIZE_16K;
    const frameIn = pcm16k.subarray(start, start + FRAME_SIZE_16K);

    resampleInto(frameIn, frame44k);            // 174 → 480  (16 kHz → 44.1 kHz)

    try {
      processor.processAudioFrame(frame44k, true);
    } catch (err) {
      const msg =
        `[denoise-worker] WASM out-of-bounds at frame ${f}/${totalFrames} ` +
        `(sample offset ${start}, segment samples ${totalSamples}, ` +
        `segment bytes ${totalSamples * 4})`;
      console.error(msg, err);
      processor.destroy();
      throw new Error(`${msg}: ${err?.message ?? err}`);
    }

    resampleInto(frame44k, frameIn);            // 480 → 174  (44.1 kHz → 16 kHz, in-place)

    if (f > 0 && f % PROGRESS_EVERY === 0) {
      self.postMessage({
        type:     "segment_progress",
        progress: Math.round((f / totalFrames) * 100),
      });
    }
  }

  processor.destroy();
  // Nullify local references so the engine can GC the WASM heap before
  // the next segment arrives.  (The WASM memory ArrayBuffer is tied to
  // the module object; dropping the reference allows collection.)
  // eslint-disable-next-line no-unused-vars
  void wasmModule;

  const elapsed = (performance.now() - t0).toFixed(0);
  console.log(
    `[denoise-worker] segment END — ${totalFrames.toLocaleString()} frames ` +
    `in ${elapsed} ms (${(totalFrames / (Number(elapsed) / 1000)).toFixed(0)} frames/s)`
  );

  return pcm16k; // modified in-place, still 16 kHz
}

// ── Message handler ───────────────────────────────────────────────────────────

self.addEventListener("message", async (e) => {
  try {
    if (e.data.type === "denoise_segment") {
      const result = await processSegment(e.data.segment);
      // Transfer zero-copy back to the main thread
      self.postMessage({ type: "segment_result", segment: result }, [result.buffer]);
    }
  } catch (err) {
    console.error("[denoise-worker] Unhandled error:", err);
    self.postMessage({ type: "error", message: err?.message ?? String(err) });
  }
});
