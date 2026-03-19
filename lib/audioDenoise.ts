/**
 * lib/audioDenoise.ts — Segmented RNNoise denoising via OfflineAudioContext
 *
 * Each 1-minute segment is rendered through an OfflineAudioContext at 44 100 Hz
 * with the NoiseSuppressorWorklet (AudioWorklet).  The worklet runs the same
 * WASM code path that ships with @timephy/rnnoise-wasm and is the only tested,
 * reliable way to call _rnnoise_process_frame without triggering WASM traps.
 *
 * Pipeline per segment:
 *   1. Upsample segment Float32Array from 16 kHz → 44 100 Hz in-browser
 *      (create AudioBuffer at 44 100 Hz with the upsampled data).
 *   2. Route it through AudioBufferSourceNode → NoiseSuppressorWorklet →
 *      ctx.destination inside an OfflineAudioContext at 44 100 Hz.
 *   3. ctx.startRendering() returns a denoised AudioBuffer at 44 100 Hz.
 *   4. Downsample result 44 100 Hz → 16 kHz (linear interpolation).
 *   5. Collect all denoised 16 kHz segments and concatenate.
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

// ── Constants ─────────────────────────────────────────────────────────────────

const TARGET_SR     = 16_000;
const WORKLET_SR    = 44_100;
const SEGMENT_SECS  = 60;
const SEGMENT_IN    = SEGMENT_SECS * TARGET_SR;   // 960 000 samples @ 16 kHz

// ── Resampling helpers ────────────────────────────────────────────────────────

/** Linear interpolation resample — returns a NEW Float32Array. */
function resample(input: Float32Array, srcRate: number, dstRate: number): Float32Array {
  if (srcRate === dstRate) return input.slice();
  const ratio  = srcRate / dstRate;
  const outLen = Math.round(input.length * dstRate / srcRate);
  const out    = new Float32Array(outLen);
  const maxSrc = input.length - 1;
  for (let i = 0; i < outLen; i++) {
    const pos  = i * ratio;
    const lo   = pos | 0;
    const hi   = lo < maxSrc ? lo + 1 : maxSrc;
    const frac = pos - lo;
    out[i]     = input[lo] + frac * (input[hi] - input[lo]);
  }
  return out;
}

// ── Single-segment processing ─────────────────────────────────────────────────

/**
 * Denoises one segment (Float32Array at 16 kHz) via OfflineAudioContext +
 * NoiseSuppressorWorklet.  Returns a new Float32Array at 16 kHz.
 */
async function processSegment(
  seg16k: Float32Array,
  segLabel: string,
): Promise<Float32Array> {
  const t0 = performance.now();

  // 1. Upsample to 44 100 Hz
  const seg44k = resample(seg16k, TARGET_SR, WORKLET_SR);

  // 2. Create OfflineAudioContext at 44 100 Hz
  const ctx = new OfflineAudioContext(1, seg44k.length, WORKLET_SR);

  // 3. Load the worklet module (each OfflineAudioContext instance needs its own load)
  await ctx.audioWorklet.addModule("/rnnoise/NoiseSuppressorWorklet.js");

  // 4. Put upsampled PCM into an AudioBuffer
  const srcBuf = ctx.createBuffer(1, seg44k.length, WORKLET_SR);
  srcBuf.copyToChannel(seg44k as Float32Array<ArrayBuffer>, 0);

  // 5. Wire: source → worklet → destination
  const source  = ctx.createBufferSource();
  source.buffer = srcBuf;

  const worklet = new AudioWorkletNode(ctx, "NoiseSuppressorWorklet");
  source.connect(worklet);
  worklet.connect(ctx.destination);
  source.start(0);

  // 6. Render (non-real-time, as fast as CPU allows)
  const rendered = await ctx.startRendering();

  // 7. Downsample 44 100 Hz → 16 kHz
  const denoised44k = rendered.getChannelData(0);
  const result16k   = resample(denoised44k, WORKLET_SR, TARGET_SR);

  const elapsed = (performance.now() - t0).toFixed(0);
  console.log(`[denoise] ${segLabel} done in ${elapsed} ms`);

  return result16k;
}

// ── Main export ───────────────────────────────────────────────────────────────

/**
 * Denoises a 16 kHz mono Float32Array using RNNoise via OfflineAudioContext.
 *
 * The audio is split into {@link SEGMENT_SECS}-second segments processed
 * sequentially.  Each segment is denoised independently; the results are
 * concatenated back into a single Float32Array at 16 kHz.
 *
 * @param pcm16k     Float32Array at 16 kHz
 * @param onProgress Called after each segment completes
 * @returns          Denoised Float32Array at 16 kHz (new allocation)
 */
export async function denoiseAudio(
  pcm16k: Float32Array,
  onProgress: (p: DenoiseProgress) => void,
): Promise<Float32Array> {
  const totalSeconds  = pcm16k.length / TARGET_SR;
  const totalSegments = Math.ceil(pcm16k.length / SEGMENT_IN);
  const results: Float32Array[] = [];
  const wallStart = performance.now();

  function calcEta(overallPct: number): number | null {
    if (overallPct < 5) return null;
    const elapsed = performance.now() - wallStart;
    return (elapsed / overallPct) * (100 - overallPct) / 1000;
  }

  for (let i = 0; i < totalSegments; i++) {
    const start  = i * SEGMENT_IN;
    const end    = Math.min(start + SEGMENT_IN, pcm16k.length);
    const seg    = pcm16k.slice(start, end);
    const label  = `segment ${i + 1}/${totalSegments}`;

    // Report start of segment
    const startPct = Math.round((i / totalSegments) * 100);
    onProgress({
      progress:         startPct,
      processedSeconds: Math.min(i * SEGMENT_SECS, totalSeconds),
      totalSeconds,
      timeRemaining:    calcEta(startPct),
      segmentIndex:     i + 1,
      totalSegments,
    });

    const denoised = await processSegment(seg, label);
    results.push(denoised);

    // Report completion of segment
    const donePct = Math.round(((i + 1) / totalSegments) * 100);
    onProgress({
      progress:         donePct,
      processedSeconds: Math.min((i + 1) * SEGMENT_SECS, totalSeconds),
      totalSeconds,
      timeRemaining:    calcEta(donePct),
      segmentIndex:     i + 1,
      totalSegments,
    });
  }

  // Concatenate all denoised segments
  const totalLen = results.reduce((s, r) => s + r.length, 0);
  const out      = new Float32Array(totalLen);
  let off        = 0;
  for (const r of results) { out.set(r, off); off += r.length; }

  console.log(
    `[denoise] All ${totalSegments} segment(s) complete — ` +
    `${(totalLen / TARGET_SR).toFixed(1)} s reconstructed.`
  );

  return out;
}
