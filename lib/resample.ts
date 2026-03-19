const TARGET_SAMPLE_RATE = 16_000;

/**
 * Decodes an audio File and resamples it to 16 kHz mono Float32Array —
 * the exact format Whisper expects.
 *
 * Console logs emitted:
 *   [resample] decoding start / end
 *   [resample] resampling start / end
 */
export async function resampleTo16kHz(file: File): Promise<Float32Array> {
  // ── 1. Decode ──────────────────────────────────────────────────────────────
  console.log(
    `[resample] Audio Decoding START — file: "${file.name}", size: ${(file.size / 1024).toFixed(1)} KB`
  );
  const t0 = performance.now();

  const arrayBuffer = await file.arrayBuffer();
  const decodeCtx = new AudioContext();
  const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
  await decodeCtx.close();

  console.log(
    `[resample] Audio Decoding END — duration: ${audioBuffer.duration.toFixed(2)} s, ` +
    `nativeRate: ${audioBuffer.sampleRate} Hz, channels: ${audioBuffer.numberOfChannels} ` +
    `(${(performance.now() - t0).toFixed(0)} ms)`
  );

  // ── 2. Resample to 16 kHz mono ─────────────────────────────────────────────
  console.log(
    `[resample] Resampling START — ${audioBuffer.sampleRate} Hz → ${TARGET_SAMPLE_RATE} Hz mono`
  );
  const t1 = performance.now();

  const numFrames = Math.ceil(audioBuffer.duration * TARGET_SAMPLE_RATE);
  const offlineCtx = new OfflineAudioContext(1, numFrames, TARGET_SAMPLE_RATE);
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start(0);

  const rendered = await offlineCtx.startRendering();
  const pcm = rendered.getChannelData(0);

  console.log(
    `[resample] Resampling END — output samples: ${pcm.length} ` +
    `(${(performance.now() - t1).toFixed(0)} ms)`
  );

  return pcm;
}
