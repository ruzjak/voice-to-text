// Singleton pipeline factory — import only inside "use client" components or workers.
// Never imported on the server (Next.js excludes it via the webpack alias).
import { pipeline, env } from "@xenova/transformers";

env.allowLocalModels = false;
env.useBrowserCache = true;

export const MODEL_NAME = "Xenova/whisper-small";
export const CHUNK_LENGTH_S = 30;
export const STRIDE_LENGTH_S = 5;
export const SAMPLE_RATE = 16_000;

export type ProgressCallback = (p: {
  status: string;
  file?: string;
  progress?: number;
}) => void;

let _instance: Awaited<ReturnType<typeof pipeline>> | null = null;

/**
 * Returns (and caches) the whisper-small ASR pipeline.
 * Subsequent calls return the cached instance immediately.
 */
export async function getTranscriber(onProgress?: ProgressCallback) {
  if (_instance) return _instance;
  _instance = await pipeline("automatic-speech-recognition", MODEL_NAME, {
    quantized: true,
    progress_callback: onProgress,
  });
  return _instance;
}

/** Call this if the model needs to be reloaded (e.g. after a hard reset). */
export function disposeTranscriber() {
  _instance = null;
}

/**
 * Splits a Float32Array (16 kHz mono) into overlapping chunks.
 * Each chunk is CHUNK_LENGTH_S seconds, with STRIDE_LENGTH_S seconds of overlap.
 */
export function buildChunks(audio: Float32Array): Float32Array[] {
  const chunkLen = CHUNK_LENGTH_S * SAMPLE_RATE;
  const step = (CHUNK_LENGTH_S - STRIDE_LENGTH_S) * SAMPLE_RATE;
  const chunks: Float32Array[] = [];
  let start = 0;
  while (start < audio.length) {
    const end = Math.min(start + chunkLen, audio.length);
    chunks.push(audio.slice(start, end));
    if (end >= audio.length) break;
    start += step;
  }
  return chunks;
}
