/**
 * Enhances an AudioBuffer in three stages using OfflineAudioContext:
 *
 *   1. High-pass filter @ 80 Hz  — strips low-frequency rumble and AC hum
 *   2. High-pass filter @ 120 Hz — second pass for stubborn low-end noise
 *   3. DynamicsCompressor        — normalises loudness, tames peaks
 *
 * Returns a new AudioBuffer at the same sample rate and channel count.
 * Does NOT mutate the original buffer.
 *
 * Console logs emitted:
 *   [enhance] START / END with timing
 */
export async function enhanceAudioBuffer(buffer: AudioBuffer): Promise<AudioBuffer> {
  console.log(
    `[enhance] Enhancement START — ${buffer.numberOfChannels} ch, ` +
    `${buffer.sampleRate} Hz, ${buffer.duration.toFixed(2)} s`
  );
  const t0 = performance.now();

  const offlineCtx = new OfflineAudioContext(
    buffer.numberOfChannels,
    buffer.length,
    buffer.sampleRate,
  );

  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;

  // ── Stage 1: High-pass @ 80 Hz ───────────────────────────────────────────
  const hp1 = offlineCtx.createBiquadFilter();
  hp1.type = "highpass";
  hp1.frequency.value = 80;
  hp1.Q.value = 0.7;

  // ── Stage 2: High-pass @ 120 Hz (second pass for stubborn rumble) ────────
  const hp2 = offlineCtx.createBiquadFilter();
  hp2.type = "highpass";
  hp2.frequency.value = 120;
  hp2.Q.value = 0.7;

  // ── Stage 3: Dynamic compression ─────────────────────────────────────────
  //   threshold : -24 dB  — start compressing at -24 dBFS
  //   knee      :  10 dB  — soft knee for natural-sounding transitions
  //   ratio     :   4:1   — moderate compression
  //   attack    :   3 ms  — fast enough to catch consonants
  //   release   : 250 ms  — smooth release avoids pumping artefacts
  const compressor = offlineCtx.createDynamicsCompressor();
  compressor.threshold.value = -24;
  compressor.knee.value      = 10;
  compressor.ratio.value     = 4;
  compressor.attack.value    = 0.003;
  compressor.release.value   = 0.25;

  source.connect(hp1);
  hp1.connect(hp2);
  hp2.connect(compressor);
  compressor.connect(offlineCtx.destination);
  source.start(0);

  const enhanced = await offlineCtx.startRendering();

  console.log(
    `[enhance] Enhancement END — ${(performance.now() - t0).toFixed(0)} ms`
  );

  return enhanced;
}
