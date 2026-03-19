/**
 * public/rnnoise/NoiseSuppressorWorklet.js
 *
 * Self-contained AudioWorklet processor for RNNoise denoising.
 * Registered via:  audioContext.audioWorklet.addModule('/rnnoise/NoiseSuppressorWorklet.js')
 *
 * Adapted from @timephy/rnnoise-wasm with import paths adjusted to match
 * the files served from /rnnoise/.  Logic is identical to the original.
 *
 * Import order matters:
 *   polyfills  — must run first: patches atob + self.location for the WASM loader
 *   rnnoise-sync — Emscripten module with embedded base64 WASM binary
 *   RnnoiseProcessor — thin adapter that wraps the Emscripten exports
 */

import "./polyfills.js";
import createRNNWasmModuleSync from "./rnnoise-sync.js";
import RnnoiseProcessor from "./RnnoiseProcessor.js";

// ── Helpers ───────────────────────────────────────────────────────────────────

function leastCommonMultiple(a, b) {
  let x = a, y = b;
  while (x !== y) { if (x > y) x -= y; else y -= x; }
  return (a * b) / x;
}

const PROCESSOR_NAME = "NoiseSuppressorWorklet";

// ── Worklet processor ─────────────────────────────────────────────────────────

class NoiseSuppressorWorklet extends AudioWorkletProcessor {
  constructor() {
    super();

    // WASM module + RNNoise context — created once per worklet instance.
    // createRNNWasmModuleSync() returns synchronously (binary is embedded
    // as base64 in rnnoise-sync.js — no network fetch required).
    this._denoiseProcessor = new RnnoiseProcessor(createRNNWasmModuleSync());

    // AudioWorklet delivers 128-sample chunks; RNNoise needs 480-sample frames.
    this._procNodeSampleRate = 128;
    this._denoiseSampleSize  = this._denoiseProcessor.getSampleLength();

    // Circular buffer sized to the LCM so roll-overs are always clean.
    this._circularBufferLength = leastCommonMultiple(
      this._procNodeSampleRate, this._denoiseSampleSize
    );
    this._circularBuffer = new Float32Array(this._circularBufferLength);

    this._inputBufferLength    = 0;
    this._denoisedBufferLength = 0;
    this._denoisedBufferIndx   = 0;
  }

  process(inputs, outputs) {
    const inData  = inputs[0][0];
    const outData = outputs[0][0];

    // Guard: no input connected / already disconnected
    if (!inData) return true;

    // Append raw chunk into circular buffer
    this._circularBuffer.set(inData, this._inputBufferLength);
    this._inputBufferLength += inData.length;

    // Denoise all complete 480-sample frames available
    for (
      ;
      this._denoisedBufferLength + this._denoiseSampleSize <= this._inputBufferLength;
      this._denoisedBufferLength += this._denoiseSampleSize
    ) {
      const frame = this._circularBuffer.subarray(
        this._denoisedBufferLength,
        this._denoisedBufferLength + this._denoiseSampleSize,
      );
      this._denoiseProcessor.processAudioFrame(frame, /* shouldDenoise */ true);
    }

    // Copy denoised samples to output when a full output chunk is ready
    const unsentLen = this._denoisedBufferIndx > this._denoisedBufferLength
      ? this._circularBufferLength - this._denoisedBufferIndx
      : this._denoisedBufferLength - this._denoisedBufferIndx;

    if (unsentLen >= outData.length) {
      outData.set(
        this._circularBuffer.subarray(
          this._denoisedBufferIndx,
          this._denoisedBufferIndx + outData.length,
        ),
        0,
      );
      this._denoisedBufferIndx += outData.length;
    }

    // Roll-over circular buffer indices at the LCM boundary
    if (this._denoisedBufferIndx === this._circularBufferLength) {
      this._denoisedBufferIndx = 0;
    }
    if (this._inputBufferLength === this._circularBufferLength) {
      this._inputBufferLength    = 0;
      this._denoisedBufferLength = 0;
    }

    return true;
  }
}

registerProcessor(PROCESSOR_NAME, NoiseSuppressorWorklet);
