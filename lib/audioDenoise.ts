export interface DenoiseProgress {
  /** 0–100 */
  progress: number;
  processedSeconds: number;
  totalSeconds: number;
  /** Estimated seconds remaining, or null while < 5 % processed. */
  timeRemaining: number | null;
}

/**
 * Denoises a 16 kHz mono Float32Array using RNNoise WASM running in a
 * dedicated Web Worker (/denoise-worker.js).
 *
 * Pipeline inside the worker:
 *   16 kHz  →  upsample to 44 100 Hz  →  RNNoise  →  downsample to 16 kHz
 *
 * @param pcm16k     Float32Array at 16 kHz (will be transferred zero-copy)
 * @param onProgress Called every ~500 RNNoise frames (≈ ~5 s of audio)
 * @returns          Cleaned Float32Array at 16 kHz
 */
export function denoiseAudio(
  pcm16k: Float32Array,
  onProgress: (p: DenoiseProgress) => void,
): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const worker = new Worker("/denoise-worker.js", { type: "module" });

    worker.addEventListener("message", (e: MessageEvent) => {
      const msg = e.data as
        | { type: "denoise_progress"; progress: number; processedSeconds: number; totalSeconds: number }
        | { type: "result"; audio: Float32Array }
        | { type: "error"; message: string };

      if (msg.type === "denoise_progress") {
        onProgress({
          progress:         msg.progress,
          processedSeconds: msg.processedSeconds,
          totalSeconds:     msg.totalSeconds,
          timeRemaining:    (msg as any).timeRemaining ?? null,
        });
      } else if (msg.type === "result") {
        worker.terminate();
        resolve(msg.audio);
      } else if (msg.type === "error") {
        worker.terminate();
        reject(new Error(msg.message));
      }
    });

    worker.addEventListener("error", (e: ErrorEvent) => {
      worker.terminate();
      reject(new Error(e.message ?? "Denoise worker crashed"));
    });

    // Transfer the buffer zero-copy into the worker
    worker.postMessage({ type: "denoise", audio: pcm16k }, [pcm16k.buffer]);
  });
}
