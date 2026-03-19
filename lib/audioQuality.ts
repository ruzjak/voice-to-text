const ANALYSIS_WINDOW_S   = 2;       // seconds to sample for analysis
const NOISE_BLOCK_MS      = 20;      // block size for noise-floor estimation
const NOISE_PERCENTILE    = 0.20;    // bottom 20 % of blocks = noise
const SNR_WARN_THRESHOLD  = 15;      // dB — below this → noisy
const RMS_WARN_THRESHOLD  = -40;     // dBFS — below this → too quiet

export interface AudioQualityResult {
  /** Overall RMS level of the 2-second sample, in dBFS */
  rmsDb: number;
  /** Estimated signal-to-noise ratio, in dB */
  snrDb: number;
}

/**
 * Analyses a 2-second window from the middle of `samples`.
 * Returns RMS (dBFS) and an SNR estimate based on noise-floor detection.
 *
 * Pure CPU — runs synchronously (fast enough for UI thread with a 2 s window).
 */
export function analyzeAudioQuality(
  samples: Float32Array,
  sampleRate: number,
): AudioQualityResult {
  // ── 1. Pick a 2-second window centred in the recording ───────────────────
  const windowLen = Math.min(samples.length, ANALYSIS_WINDOW_S * sampleRate);
  const offset    = Math.floor((samples.length - windowLen) / 2);
  const win       = samples.subarray(offset, offset + windowLen);

  // ── 2. Overall RMS → dBFS ────────────────────────────────────────────────
  let sumSq = 0;
  for (let i = 0; i < win.length; i++) sumSq += win[i] * win[i];
  const rms   = Math.sqrt(sumSq / win.length);
  const rmsDb = 20 * Math.log10(Math.max(rms, 1e-10));

  // ── 3. Noise-floor estimate ───────────────────────────────────────────────
  // Divide the window into 20 ms blocks; the quietest N % are treated as
  // background noise. SNR = 20·log10(signal_rms / noise_rms).
  const blockSize = Math.max(1, Math.floor((NOISE_BLOCK_MS / 1000) * sampleRate));
  const blockRms: number[] = [];

  for (let i = 0; i + blockSize <= win.length; i += blockSize) {
    let bSq = 0;
    for (let j = i; j < i + blockSize; j++) bSq += win[j] * win[j];
    blockRms.push(Math.sqrt(bSq / blockSize));
  }

  blockRms.sort((a, b) => a - b);
  const noiseCount = Math.max(1, Math.floor(blockRms.length * NOISE_PERCENTILE));

  let noiseRmsSq = 0;
  for (let i = 0; i < noiseCount; i++) noiseRmsSq += blockRms[i] * blockRms[i];
  const noiseRms = Math.sqrt(noiseRmsSq / noiseCount);

  const snrDb =
    noiseRms < 1e-10
      ? 60  // effectively silent noise floor
      : 20 * Math.log10(Math.max(rms / noiseRms, 1));

  return { rmsDb, snrDb };
}

/** Returns true when the recording quality falls below safe thresholds. */
export function isQualityPoor({ rmsDb, snrDb }: AudioQualityResult): boolean {
  return snrDb < SNR_WARN_THRESHOLD || rmsDb < RMS_WARN_THRESHOLD;
}
