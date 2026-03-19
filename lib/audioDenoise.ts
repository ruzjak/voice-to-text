/**
 * lib/audioDenoise.ts — Segmented RNNoise denoising orchestrator
 *
 * Splits a 16 kHz Float32Array into 1-minute segments, sends them one at a
 * time to /denoise-worker.js via Transferable Objects, collects the results,
 * and stitches them back into a single buffer.
 *
 * Processing one segment at a time keeps peak WASM heap usage at ~3.8 MB
 * (one 60-second slice × 4 bytes/sample) regardless of total audio length.
 */

export interface DenoiseProgress {
  /** Overall 0–100 across all segments. */
  progress: number;
  processedSeconds: number;
  totalSeconds: number;
  /** Estimated seconds remaining, or null while < 5 % processed. */
  timeRemaining: number | null;
  /** 1-based index of the segment currently being processed. */
  segmentIndex: number;
  /** Total number of segments the audio was split into. */
  totalSegments: number;
}

// ── Segmentation constants ────────────────────────────────────────────────────

/** Audio seconds per segment sent to the worker. */
const SEGMENT_SECS    = 60;
/** Samples per segment at 16 kHz. ~3.84 MB each — well within WASM limits. */
const SEGMENT_SAMPLES = SEGMENT_SECS * 16_000;   // 960 000
const TARGET_SR       = 16_000;

function splitSegments(pcm: Float32Array): Float32Array[] {
  const segs: Float32Array[] = [];
  for (let off = 0; off < pcm.length; off += SEGMENT_SAMPLES) {
    // slice() copies — this frees the source region from the transferred buffer
    segs.push(pcm.slice(off, Math.min(off + SEGMENT_SAMPLES, pcm.length)));
  }
  return segs;
}

// ── Main export ───────────────────────────────────────────────────────────────

/**
 * Denoises a 16 kHz mono Float32Array using RNNoise WASM running in a
 * dedicated Web Worker (/denoise-worker.js).
 *
 * The audio is split into {@link SEGMENT_SECS}-second segments. Each segment
 * is transferred zero-copy to the worker, processed, and transferred back
 * before the next segment is sent, preventing the WASM heap from accumulating
 * the full recording in memory.
 *
 * @param pcm16k     Float32Array at 16 kHz (copied into segments internally)
 * @param onProgress Called on every worker progress message
 * @returns          Denoised Float32Array at 16 kHz (new allocation)
 */
export function denoiseAudio(
  pcm16k: Float32Array,
  onProgress: (p: DenoiseProgress) => void,
): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const totalSeconds  = pcm16k.length / TARGET_SR;
    const segments      = splitSegments(pcm16k);
    const totalSegments = segments.length;
    const results: Float32Array[] = [];
    let segmentIndex    = 0;
    const wallStart     = performance.now();

    // Heap-size advisory logged to the console
    const totalMB = (pcm16k.length * 4) / (1024 * 1024);
    if (totalMB > 50) {
      console.warn(
        `[denoise] Input is ${totalMB.toFixed(0)} MB total. ` +
        `Splitting into ${totalSegments} × ${SEGMENT_SECS} s segment(s) ` +
        `(≈${(SEGMENT_SAMPLES * 4 / 1024 / 1024).toFixed(1)} MB each).`
      );
    }

    const worker = new Worker("/denoise-worker.js", { type: "module" });

    /** Compute overall ETA from wall-clock elapsed and progress fraction. */
    function calcEta(overallPct: number): number | null {
      if (overallPct < 5) return null;
      const elapsed = performance.now() - wallStart;
      return (elapsed / overallPct) * (100 - overallPct) / 1000;
    }

    /** Send the next segment, or finalise if all are done. */
    function sendNext(): void {
      if (segmentIndex >= totalSegments) {
        worker.terminate();

        // Reconstruct into a single contiguous Float32Array
        const totalLen = results.reduce((s, r) => s + r.length, 0);
        const out = new Float32Array(totalLen);
        let off = 0;
        for (const r of results) { out.set(r, off); off += r.length; }

        console.log(
          `[denoise] All ${totalSegments} segment(s) complete — ` +
          `${(totalLen / TARGET_SR).toFixed(1)} s reconstructed.`
        );
        resolve(out);
        return;
      }

      const seg   = segments[segmentIndex];
      const segMB = (seg.length * 4) / (1024 * 1024);
      if (segMB > 50) {
        console.warn(
          `[denoise] Segment ${segmentIndex + 1}/${totalSegments} ` +
          `is ${segMB.toFixed(1)} MB — may stress WASM heap.`
        );
      }

      console.log(
        `[denoise] → Sending segment ${segmentIndex + 1}/${totalSegments} ` +
        `(${seg.length.toLocaleString()} samples, ${segMB.toFixed(2)} MB)`
      );

      // Transfer zero-copy — `seg.buffer` becomes detached in this thread
      worker.postMessage({ type: "denoise_segment", segment: seg }, [seg.buffer]);
    }

    worker.addEventListener("message", (e: MessageEvent) => {
      const msg = e.data as
        | { type: "segment_progress"; progress: number }
        | { type: "segment_result";   segment: Float32Array }
        | { type: "error";            message: string };

      if (msg.type === "segment_progress") {
        // Blend intra-segment progress into the overall 0–100 scale
        const overall = Math.round(
          ((segmentIndex + msg.progress / 100) / totalSegments) * 100
        );
        const processedSeconds = Math.min(
          segmentIndex * SEGMENT_SECS +
            (msg.progress / 100) * (segments[segmentIndex]?.length ?? 0) / TARGET_SR,
          totalSeconds,
        );
        onProgress({
          progress:         overall,
          processedSeconds,
          totalSeconds,
          timeRemaining:    calcEta(overall),
          segmentIndex:     segmentIndex + 1,
          totalSegments,
        });

      } else if (msg.type === "segment_result") {
        results.push(msg.segment);
        segmentIndex++;

        const overall          = Math.round((segmentIndex / totalSegments) * 100);
        const processedSeconds = Math.min(segmentIndex * SEGMENT_SECS, totalSeconds);
        onProgress({
          progress:         overall,
          processedSeconds,
          totalSeconds,
          timeRemaining:    calcEta(overall),
          segmentIndex,
          totalSegments,
        });

        sendNext();

      } else if (msg.type === "error") {
        worker.terminate();
        reject(new Error(msg.message));
      }
    });

    worker.addEventListener("error", (e: ErrorEvent) => {
      worker.terminate();
      reject(new Error(e.message ?? "Denoise worker crashed"));
    });

    sendNext();
  });
}
