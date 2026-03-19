/**
 * public/denoise-worker.js — RNNoise offline denoising Web Worker
 *
 * Loaded as a module worker: new Worker('/denoise-worker.js', { type: 'module' })
 * The rnnoise-sync.js is served from /rnnoise/ (WASM binary embedded as base64).
 *
 * ── Memory-safe pipeline ──────────────────────────────────────────────────────
 * Previous approach: bulk-upsample entire buffer to 44100 Hz → OOM for long audio.
 *   60-min @ 16 kHz = 57.6 M samples → ×2.75 = 158.8 M samples = 635 MB just for
 *   the intermediate buffer.  WASM heap immediately exceeds its limits.
 *
 * New approach: PER-FRAME resampling.
 *   One RNNoise frame = 480 samples @ 44 100 Hz = 10 ms.
 *   At 16 000 Hz that equals:  480 × (16000 / 44100) = 173.87 ≈ 174 samples.
 *   For each frame we:
 *     1. Read 174 samples from pcm16k (via subarray — zero-copy view).
 *     2. Upsample those 174 samples → 480-sample buffer (pre-allocated, reused).
 *     3. Run RNNoise on the 480-sample buffer (in-place).
 *     4. Downsample 480 samples → 174 samples back into pcm16k (in-place).
 *   Peak extra memory: one Float32Array(480) = 1 920 bytes + tiny WASM heap.
 *   The input buffer pcm16k is modified in-place; no second large buffer needed.
 *
 * Message protocol
 * ────────────────
 * IN  { type: 'denoise', audio: Float32Array }   (transferable, 16 kHz mono)
 *
 * OUT { type: 'denoise_progress', progress, processedSeconds, totalSeconds }
 * OUT { type: 'result',           audio: Float32Array }   (16 kHz mono, in-place)
 * OUT { type: 'error',            message: string }
 */

import createRNNWasmModuleSync from "/rnnoise/rnnoise-sync.js";

// ── Constants ─────────────────────────────────────────────────────────────────

/** RNNoise native sample rate (Hz). Input MUST be at this rate at the WASM boundary. */
const RNNOISE_SR         = 44100;
/** Target sample rate for Whisper (Hz). Input arrives and output leaves at this rate. */
const TARGET_SR          = 16000;
/** Exact samples per RNNoise frame at RNNOISE_SR. Non-negotiable — WASM crashes otherwise. */
const RNNOISE_FRAME_SIZE = 480;
/** Bytes occupied by one WASM frame buffer (float32). */
const RNNOISE_FRAME_BYTES = RNNOISE_FRAME_SIZE * 4;
/** int16 scale factor — RnnoiseProcessor multiplies/divides by this internally. */
const SHIFT = 32768;

/**
 * How many TARGET_SR samples correspond to one RNNOISE_FRAME_SIZE frame.
 * = 480 × (16000 / 44100) = 173.87 → rounded to 174.
 * Using Math.round minimises cumulative timing drift (< 3 s over 60 min).
 */
const FRAME_SIZE_16K = Math.round(RNNOISE_FRAME_SIZE * TARGET_SR / RNNOISE_SR); // 174

/** Post a progress message every N frames (≈ every 18.7 s of audio). */
const PROGRESS_EVERY = 1720; // 1720 × 174 / 16000 ≈ 18.7 s per update

// ── ETA helper ────────────────────────────────────────────────────────────────
//
// Returns remaining seconds as a number, or null during the first 5 % of
// frames (< PROGRESS_EVERY frames processed) when the estimate would be noisy.

function calcEta(framesDone, totalFrames, elapsedMs) {
  if (framesDone < PROGRESS_EVERY || totalFrames === 0) return null;
  const msPerFrame   = elapsedMs / framesDone;
  const framesLeft   = totalFrames - framesDone;
  return (msPerFrame * framesLeft) / 1000; // seconds
}

// ── In-place linear resampler ─────────────────────────────────────────────────
//
// Resamples `input` (length A) into `output` (length B) using linear
// interpolation.  Both buffers are caller-supplied — no allocation inside.
// Works correctly for both up-sampling (A < B) and down-sampling (A > B).

function resampleInto(input, output) {
  const srcLen = input.length;
  const dstLen = output.length;
  const ratio  = srcLen / dstLen;          // src samples per dst sample
  const maxSrc = srcLen - 1;

  for (let i = 0; i < dstLen; i++) {
    const pos  = i * ratio;
    const lo   = pos | 0;                  // Math.floor, faster
    const hi   = lo < maxSrc ? lo + 1 : maxSrc;
    const frac = pos - lo;
    output[i]  = input[lo] + frac * (input[hi] - input[lo]);
  }
}

// ── RnnoiseProcessor (inlined from @timephy/rnnoise-wasm/dist/RnnoiseProcessor) ─
//
// The package does not export this class via the package.json exports map, so we
// inline the ~50-line implementation here.  Logic is identical to the original.

class RnnoiseProcessor {
  /**
   * @param {object} wasmInterface — Emscripten module returned by createRNNWasmModuleSync()
   */
  constructor(wasmInterface) {
    this._wasmInterface = wasmInterface;
    // Allocate the WASM-side input/output buffer (tiny — 480 × 4 = 1 920 bytes)
    this._wasmPcmInput = wasmInterface._malloc(RNNOISE_FRAME_BYTES);
    if (!this._wasmPcmInput) {
      throw new Error(
        `RNNoise WASM _malloc(${RNNOISE_FRAME_BYTES}) returned 0 — ` +
        "WASM heap may be exhausted."
      );
    }
    // Float32Array index = byte pointer >> 2 (each f32 is 4 bytes)
    this._inputF32Idx = this._wasmPcmInput >> 2;
    this._context     = wasmInterface._rnnoise_create();
    this._destroyed   = false;
  }

  /**
   * Process one 480-sample frame.
   * When shouldDenoise=true the denoised PCM is written back into `frame`.
   * @param {Float32Array} frame        — exactly RNNOISE_FRAME_SIZE samples, range [-1, 1]
   * @param {boolean}      shouldDenoise
   * @returns {number} VAD score 0–1
   */
  processAudioFrame(frame, shouldDenoise = false) {
    // Scale [-1,1] → [-32768, 32768] and copy into WASM heap
    for (let i = 0; i < RNNOISE_FRAME_SIZE; i++) {
      this._wasmInterface.HEAPF32[this._inputF32Idx + i] = frame[i] * SHIFT;
    }

    // Run the neural network (in-place on the WASM buffer)
    const vad = this._wasmInterface._rnnoise_process_frame(
      this._context,
      this._wasmPcmInput,
      this._wasmPcmInput,
    );

    // Scale back and write denoised PCM into the caller's buffer
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

// ── Main denoising function ───────────────────────────────────────────────────

async function denoiseAudio(pcm16k) {
  const totalSamples = pcm16k.length;
  const totalSeconds = totalSamples / TARGET_SR;
  const totalFrames  = Math.floor(totalSamples / FRAME_SIZE_16K);

  console.log(
    `[denoise] START — ${totalSeconds.toFixed(1)} s, ${totalSamples.toLocaleString()} samples ` +
    `→ ${totalFrames.toLocaleString()} frames\n` +
    `          FRAME_SIZE_16K=${FRAME_SIZE_16K}, RNNOISE_FRAME_SIZE=${RNNOISE_FRAME_SIZE}, ` +
    `RNNOISE_SR=${RNNOISE_SR} Hz`
  );

  // ── 1. Init WASM (synchronous — binary is embedded as base64) ─────────────
  console.log("[denoise] Initialising RNNoise WASM module…");
  const wasmModule = createRNNWasmModuleSync();

  let processor;
  try {
    processor = new RnnoiseProcessor(wasmModule);
  } catch (err) {
    console.error(
      `[denoise] RnnoiseProcessor init failed — ` +
      `total input samples: ${totalSamples.toLocaleString()}, ` +
      `input buffer bytes: ${totalSamples * 4}:`,
      err
    );
    throw err;
  }

  // ── 2. Pre-allocate the single reusable 44.1 kHz frame buffer ─────────────
  // This is the ONLY extra allocation in the hot loop — 480 × 4 = 1 920 bytes.
  const frame44k = new Float32Array(RNNOISE_FRAME_SIZE);

  console.log(
    `[denoise] Processing — reusing one ${RNNOISE_FRAME_SIZE}-sample (${RNNOISE_FRAME_BYTES} B) ` +
    "frame buffer per iteration (no heap growth in loop)"
  );

  // Report 0 % so the UI shows the progress bar immediately
  self.postMessage({ type: "denoise_progress", progress: 0, processedSeconds: 0, totalSeconds });

  const t0 = performance.now();

  // ── 3. Per-frame loop ──────────────────────────────────────────────────────
  // Each iteration:
  //   a) View 174 samples of pcm16k via subarray (zero-copy, no allocation)
  //   b) Upsample 174 → 480 into frame44k (pre-allocated, reused)
  //   c) RNNoise on frame44k in-place
  //   d) Downsample 480 → 174 back into pcm16k via the subarray view (in-place)
  for (let f = 0; f < totalFrames; f++) {
    const start   = f * FRAME_SIZE_16K;
    const frameIn = pcm16k.subarray(start, start + FRAME_SIZE_16K); // view, no copy

    // (a) Upsample 16 kHz slice → 44.1 kHz frame (written into pre-allocated buffer)
    resampleInto(frameIn, frame44k);

    // (b) RNNoise — catch WASM traps and log context for diagnosis
    try {
      processor.processAudioFrame(frame44k, /* shouldDenoise */ true);
    } catch (err) {
      const msg =
        `[denoise] WASM out-of-bounds at frame ${f}/${totalFrames} ` +
        `(sample offset ${start}, total input samples ${totalSamples}, ` +
        `input buffer bytes ${totalSamples * 4})`;
      console.error(msg, err);
      processor.destroy();
      throw new Error(`${msg}: ${err?.message ?? err}`);
    }

    // (c) Downsample 44.1 kHz frame → 16 kHz slice (writes back into pcm16k in-place)
    resampleInto(frame44k, frameIn);

    // (d) Periodic progress report
    if (f > 0 && f % PROGRESS_EVERY === 0) {
      const processedSeconds = (start + FRAME_SIZE_16K) / TARGET_SR;
      const timeRemaining    = calcEta(f, totalFrames, performance.now() - t0);
      self.postMessage({
        type: "denoise_progress",
        progress:         Math.round((f / totalFrames) * 100),
        processedSeconds: Math.min(processedSeconds, totalSeconds),
        totalSeconds,
        timeRemaining,  // seconds | null
      });
    }
  }

  processor.destroy();

  const elapsed = (performance.now() - t0).toFixed(0);
  console.log(
    `[denoise] END — ${totalFrames.toLocaleString()} frames in ${elapsed} ms ` +
    `(${(totalFrames / (Number(elapsed) / 1000)).toFixed(0)} frames/s)`
  );

  // Return the input buffer modified in-place — still at 16 kHz, no extra allocation.
  return pcm16k;
}

// ── Message handler ───────────────────────────────────────────────────────────

self.addEventListener("message", async (e) => {
  try {
    if (e.data.type === "denoise") {
      const cleaned = await denoiseAudio(e.data.audio);
      // Transfer the buffer zero-copy back to the main thread
      self.postMessage({ type: "result", audio: cleaned }, [cleaned.buffer]);
    }
  } catch (err) {
    console.error("[denoise] Unhandled error:", err);
    self.postMessage({ type: "error", message: err?.message ?? String(err) });
  }
});
