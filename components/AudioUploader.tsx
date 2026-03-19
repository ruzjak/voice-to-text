"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import WaveformCanvas from "./WaveformCanvas";
import ExportButtons from "./ExportButtons";
import { resampleTo16kHz, resampleBufferTo16kHz } from "@/lib/resample";
import { analyzeAudioQuality, isQualityPoor } from "@/lib/audioQuality";
import { enhanceAudioBuffer } from "@/lib/audioEnhance";
import { denoiseAudio } from "@/lib/audioDenoise";

const ACCEPTED_EXTENSIONS = [".mp3", ".m4a", ".wav"];

interface AudioFile {
  file: File;
  name: string;
  size: number;
  url: string;
  duration: number;
  channelData: Float32Array;
  /** Full decoded buffer — kept for quality analysis and enhancement. */
  rawBuffer: AudioBuffer;
}

type QualityState =
  | { status: "idle" }
  | { status: "good";      rmsDb: number; snrDb: number }
  | { status: "warn";      rmsDb: number; snrDb: number }
  | { status: "enhancing"; progress: number; processedSeconds: number; totalSeconds: number; timeRemaining: number | null; segmentIndex: number; totalSegments: number }
  | { status: "enhanced";  rmsDb: number; snrDb: number };

type TranscriptState =
  | { status: "idle" }
  | { status: "loading-model"; file: string; progress: number; device: "webgpu" | "wasm" | null }
  | { status: "transcribing"; current: number; total: number; progress: number; partialText: string; statusLabel: string; processedSeconds: number; totalSeconds: number; timeRemaining: number | null; device: "webgpu" | "wasm" | null }
  | { status: "done"; text: string }
  | { status: "error"; message: string };

// Messages sent FROM public/worker.js TO the main thread
type WorkerMessage =
  | { type: "status"; status: "loading" | "ready" }
  | { type: "device"; device: "webgpu" | "wasm" }
  | { type: "model_progress"; file: string; progress: number }
  | { type: "chunk_progress"; current: number; total: number; progress: number; partialText: string; statusLabel: string; processedSeconds: number; totalSeconds: number; timeRemaining: number | null }
  | { type: "result"; text: string }
  | { type: "error"; message: string };

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** Formats a raw seconds value as MM:SS (e.g. 322 → "05:22"). */
function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export default function AudioUploader() {
  const [audioFile, setAudioFile] = useState<AudioFile | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [transcriptState, setTranscriptState] = useState<TranscriptState>({ status: "idle" });
  const [qualityState, setQualityState] = useState<QualityState>({ status: "idle" });

  // ── Comparison / source-selection state ───────────────────────────────────
  /** Quality metrics captured immediately after file upload (on original audio). */
  const [originalQuality, setOriginalQuality]     = useState<{ rmsDb: number; snrDb: number } | null>(null);
  /** Quality metrics captured after the full enhance + denoise pipeline. */
  const [enhancedQuality, setEnhancedQuality]     = useState<{ rmsDb: number; snrDb: number } | null>(null);
  /** Which audio source will be fed to Whisper. Auto-switches to "enhanced" when done. */
  const [transcriptionSource, setTranscriptionSource] = useState<"original" | "enhanced">("original");
  /** True when the enhanced audio still falls below the SNR quality threshold. */
  const [showQualityToast, setShowQualityToast]   = useState(false);

  const deviceRef       = useRef<"webgpu" | "wasm" | null>(null);
  // Cleaned 16 kHz PCM ready for Whisper. Set after full enhance+denoise pipeline.
  const enhancedPcmRef  = useRef<Float32Array | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const rafRef = useRef<number | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // ── Worker lifecycle ──────────────────────────────────────────────────────

  const getOrCreateWorker = useCallback((): Worker => {
    if (workerRef.current) return workerRef.current;

    // public/worker.js is served at /worker.js — runs as a module worker
    // completely outside the webpack bundle, keeping the main thread free.
    const worker = new Worker("/worker.js", { type: "module" });

    worker.addEventListener("message", (e: MessageEvent<WorkerMessage>) => {
      const msg = e.data;

      if (msg.type === "status" && msg.status === "loading") {
        setTranscriptState({ status: "loading-model", file: "", progress: 0, device: deviceRef.current });

      } else if (msg.type === "device") {
        deviceRef.current = msg.device;
        // Patch device into whichever state is current
        setTranscriptState((prev) =>
          prev.status === "loading-model" ? { ...prev, device: msg.device } : prev
        );

      } else if (msg.type === "model_progress") {
        setTranscriptState((prev) => ({
          status: "loading-model",
          file: msg.file,
          progress: msg.progress,
          device: prev.status === "loading-model" ? prev.device : deviceRef.current,
        }));

      } else if (msg.type === "chunk_progress") {
        setTranscriptState({
          status: "transcribing",
          current: msg.current,
          total: msg.total,
          progress: msg.progress,
          partialText: msg.partialText,
          statusLabel: msg.statusLabel,
          processedSeconds: msg.processedSeconds,
          totalSeconds: msg.totalSeconds,
          timeRemaining: msg.timeRemaining,
          device: deviceRef.current,
        });

      } else if (msg.type === "result") {
        setTranscriptState({ status: "done", text: msg.text });

      } else if (msg.type === "error") {
        setTranscriptState({ status: "error", message: msg.message });
      }
    });

    worker.addEventListener("error", (e) => {
      setTranscriptState({ status: "error", message: e.message });
    });

    workerRef.current = worker;
    return worker;
  }, []);

  // ── Transcription entry point ─────────────────────────────────────────────

  const handleTranscribe = useCallback(async () => {
    if (!audioFile) return;
    setTranscriptState({ status: "loading-model", file: "", progress: 0, device: deviceRef.current });

    try {
      let audio: Float32Array;
      if (transcriptionSource === "enhanced" && enhancedPcmRef.current) {
        // Already 16 kHz + Web-Audio filtered + RNNoise denoised — use directly.
        // Slice so the buffer stays reusable for subsequent Transcribe clicks.
        console.log("[app] Using enhanced PCM — skipping resample");
        audio = enhancedPcmRef.current.slice();
      } else {
        console.log(`[app] Resampling original audio to 16 kHz START (source: ${transcriptionSource})`);
        audio = await resampleTo16kHz(audioFile.file);
        console.log(`[app] Resampling to 16 kHz END — ${audio.length} samples`);
      }

      const worker = getOrCreateWorker();

      // Send load + transcribe back-to-back.
      // The worker's getOrLoadModel() ensures transcribe always awaits the
      // model promise, so there is no race condition regardless of timing.
      worker.postMessage({ type: "load" });
      worker.postMessage({ type: "transcribe", audio }, [audio.buffer]);
    } catch (err) {
      setTranscriptState({
        status: "error",
        message: err instanceof Error ? err.message : "Resampling failed",
      });
    }
  }, [audioFile, transcriptionSource, getOrCreateWorker]);

  // ── Audio enhancement ────────────────────────────────────────────────────

  const handleEnhance = useCallback(async () => {
    if (!audioFile?.rawBuffer) return;

    // Kick off with "enhancing" at 0 %
    setQualityState({ status: "enhancing", progress: 0, processedSeconds: 0, totalSeconds: 0, timeRemaining: null, segmentIndex: 0, totalSegments: 0 });
    enhancedPcmRef.current = null;

    try {
      // ── Stage 1: Web Audio filter chain (fast, no progress needed) ────────
      console.log("[enhance] Stage 1 — Web Audio filter chain");
      const filteredBuffer = await enhanceAudioBuffer(audioFile.rawBuffer);

      // ── Stage 2: Resample filtered buffer to 16 kHz ───────────────────────
      console.log("[enhance] Stage 2 — Resample to 16 kHz");
      const pcm16k = await resampleBufferTo16kHz(filteredBuffer);

      // ── Stage 3: RNNoise denoising (slow — show progress bar) ─────────────
      console.log("[enhance] Stage 3 — RNNoise denoising");
      const denoised = await denoiseAudio(pcm16k, (p) => {
        setQualityState({
          status:           "enhancing",
          progress:         p.progress,
          processedSeconds: p.processedSeconds,
          totalSeconds:     p.totalSeconds,
          timeRemaining:    p.timeRemaining,
          segmentIndex:     p.segmentIndex,
          totalSegments:    p.totalSegments,
        });
      });

      enhancedPcmRef.current = denoised;

      // Re-measure quality on the cleaned audio
      const result = analyzeAudioQuality(denoised, 16_000);
      console.log(
        `[enhance] Post-denoise RMS: ${result.rmsDb.toFixed(1)} dBFS, ` +
        `SNR: ${result.snrDb.toFixed(1)} dB`
      );
      setEnhancedQuality({ rmsDb: result.rmsDb, snrDb: result.snrDb });
      setQualityState({ status: "enhanced", rmsDb: result.rmsDb, snrDb: result.snrDb });

      // Promote the enhanced buffer only when it actually improves clarity.
      // If the denoiser made things worse (heavy distortion artefacts), keep
      // the original selected so the user doesn't accidentally transcribe
      // degraded audio without noticing.
      const isDegraded = originalQuality !== null && result.snrDb < originalQuality.snrDb;
      if (!isDegraded) {
        setTranscriptionSource("enhanced");
        console.log("[enhance] Promoted to enhanced — SNR improved.");
      } else {
        console.warn(
          `[enhance] Enhancement degraded SNR ` +
          `(${originalQuality!.snrDb.toFixed(1)} → ${result.snrDb.toFixed(1)} dB). ` +
          "Keeping original as active source."
        );
      }

      // If the enhanced SNR is still below the safe threshold, warn the user
      if (isQualityPoor(result)) setShowQualityToast(true);

    } catch (err) {
      console.error("[enhance] Error:", err);
      setQualityState((prev) =>
        prev.status === "enhancing" ? { status: "warn", rmsDb: 0, snrDb: 0 } : prev
      );
    }
  }, [audioFile, originalQuality]);

  // ── File processing ───────────────────────────────────────────────────────

  const processFile = useCallback(async (file: File) => {
    setUploadError(null);
    setAudioFile(null);
    setIsPlaying(false);
    setCurrentTime(0);
    setTranscriptState({ status: "idle" });
    setQualityState({ status: "idle" });
    setOriginalQuality(null);
    setEnhancedQuality(null);
    setTranscriptionSource("original");
    setShowQualityToast(false);
    enhancedPcmRef.current = null;

    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(`.${ext}`)) {
      setUploadError("Unsupported file type. Please upload .mp3, .m4a, or .wav files.");
      return;
    }

    setLoading(true);
    try {
      console.log(`[app] Audio Decoding START — "${file.name}"`);
      const t0 = performance.now();
      const arrayBuffer = await file.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
      await audioCtx.close();
      console.log(
        `[app] Audio Decoding END — ${audioBuffer.duration.toFixed(2)} s, ` +
        `${audioBuffer.sampleRate} Hz (${(performance.now() - t0).toFixed(0)} ms)`
      );

      const rawData = audioBuffer.getChannelData(0);
      const samples = 4000;
      const blockSize = Math.floor(rawData.length / samples);
      const channelData = new Float32Array(samples);
      for (let i = 0; i < samples; i++) {
        let sum = 0;
        for (let j = 0; j < blockSize; j++) {
          sum += Math.abs(rawData[i * blockSize + j]);
        }
        channelData[i] = sum / blockSize;
      }

      setAudioFile({
        file,
        name: file.name,
        size: file.size,
        url: URL.createObjectURL(file),
        duration: audioBuffer.duration,
        channelData,
        rawBuffer: audioBuffer,
      });

      // ── Pre-flight quality check (synchronous, < 5 ms for a 2 s window) ──
      const rawSamples  = audioBuffer.getChannelData(0);
      const qualityResult = analyzeAudioQuality(rawSamples, audioBuffer.sampleRate);
      console.log(
        `[quality] RMS: ${qualityResult.rmsDb.toFixed(1)} dBFS, ` +
        `SNR: ${qualityResult.snrDb.toFixed(1)} dB`
      );
      setOriginalQuality({ rmsDb: qualityResult.rmsDb, snrDb: qualityResult.snrDb });
      setQualityState({
        status: isQualityPoor(qualityResult) ? "warn" : "good",
        rmsDb:  qualityResult.rmsDb,
        snrDb:  qualityResult.snrDb,
      });
    } catch {
      setUploadError("Failed to decode audio. The file may be corrupted or unsupported.");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      processFile(files[0]);
    },
    [processFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  // ── Playback ──────────────────────────────────────────────────────────────

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    } else {
      audio.play();
      const tick = () => {
        setCurrentTime(audio.currentTime);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    }
    setIsPlaying(!isPlaying);
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  };

  const handleSeek = (ratio: number) => {
    const audio = audioRef.current;
    if (!audio || !audioFile) return;
    audio.currentTime = ratio * audioFile.duration;
    setCurrentTime(audio.currentTime);
  };

  const handleReset = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (audioFile) URL.revokeObjectURL(audioFile.url);
    setAudioFile(null);
    setIsPlaying(false);
    setCurrentTime(0);
    setUploadError(null);
    setTranscriptState({ status: "idle" });
    setQualityState({ status: "idle" });
    setOriginalQuality(null);
    setEnhancedQuality(null);
    setTranscriptionSource("original");
    setShowQualityToast(false);
    enhancedPcmRef.current = null;
    if (inputRef.current) inputRef.current.value = "";
  };

  const waveformProgress = audioFile ? currentTime / audioFile.duration : 0;
  const isBusy =
    transcriptState.status === "loading-model" ||
    transcriptState.status === "transcribing";

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      {!audioFile && (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onClick={() => inputRef.current?.click()}
          className={`
            relative flex flex-col items-center justify-center gap-4
            rounded-2xl border-2 border-dashed cursor-pointer
            px-8 py-16 transition-all duration-200
            ${isDragging
              ? "border-violet-400 bg-violet-500/10 scale-[1.01]"
              : "border-gray-700 bg-gray-900/50 hover:border-violet-500 hover:bg-gray-900"
            }
            ${loading ? "pointer-events-none opacity-60" : ""}
          `}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS.join(",")}
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          {loading ? (
            <>
              <div className="w-12 h-12 rounded-full border-4 border-violet-500 border-t-transparent animate-spin" />
              <p className="text-gray-400 text-sm">Decoding audio…</p>
            </>
          ) : (
            <>
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-gray-800 border border-gray-700">
                <svg className="w-8 h-8 text-violet-400" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
              </div>
              <div className="text-center space-y-1">
                <p className="text-white font-medium">Drop your audio file here</p>
                <p className="text-gray-500 text-sm">or click to browse — .mp3, .m4a, .wav</p>
              </div>
            </>
          )}
        </div>
      )}

      {/* Upload error */}
      {uploadError && (
        <div className="flex items-center gap-3 rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-400">
          <svg className="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-.75-4.75a.75.75 0 001.5 0v-4.5a.75.75 0 00-1.5 0v4.5zm.75-7.5a.75.75 0 110 1.5.75.75 0 010-1.5z" clipRule="evenodd" />
          </svg>
          {uploadError}
        </div>
      )}

      {/* Player card */}
      {audioFile && (
        <div className="rounded-2xl bg-gray-900 border border-gray-800 overflow-hidden">
          {/* File info */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-violet-500/20 shrink-0">
                <svg className="w-5 h-5 text-violet-400" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                </svg>
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="text-white font-medium text-sm truncate">{audioFile.name}</p>
                  {/* Status Badge */}
                  {enhancedQuality ? (
                    <span className="shrink-0 inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium
                      bg-violet-500/15 text-violet-300 border border-violet-500/30">
                      ✨ Enhanced
                    </span>
                  ) : (
                    <span className="shrink-0 text-xs px-2 py-0.5 rounded-full font-medium
                      bg-gray-700/60 text-gray-400 border border-gray-600/50">
                      Original
                    </span>
                  )}
                </div>
                <p className="text-gray-500 text-xs">
                  {formatSize(audioFile.size)} · {formatDuration(audioFile.duration)}
                </p>
              </div>
            </div>
            <button
              onClick={handleReset}
              className="text-gray-500 hover:text-gray-300 transition-colors p-1 rounded-lg hover:bg-gray-800 shrink-0"
              title="Remove file"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Waveform */}
          <div className="px-5 pt-5 pb-2">
            <WaveformCanvas
              channelData={audioFile.channelData}
              progress={waveformProgress}
              onSeek={handleSeek}
            />
          </div>

          {/* Transport */}
          <div className="flex items-center justify-between px-5 py-4">
            <span className="text-gray-500 text-xs tabular-nums">{formatDuration(currentTime)}</span>
            <button
              onClick={togglePlay}
              className="flex items-center justify-center w-11 h-11 rounded-full bg-violet-600 hover:bg-violet-500 active:scale-95 transition-all shadow-lg shadow-violet-900/40"
            >
              {isPlaying ? (
                <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-white translate-x-0.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
            </button>
            <span className="text-gray-500 text-xs tabular-nums">{formatDuration(audioFile.duration)}</span>
          </div>

          {/* Transcribe button */}
          <div className="px-5 pb-5">
            <button
              onClick={handleTranscribe}
              disabled={isBusy}
              className="w-full flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-medium transition-all
                bg-violet-600 hover:bg-violet-500 active:scale-[0.99] text-white
                disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-violet-600"
            >
              {isBusy ? (
                <span className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                </svg>
              )}
              {isBusy
                ? "Working…"
                : transcriptState.status === "done"
                ? `Transcribe again · ${transcriptionSource === "enhanced" ? "Enhanced" : "Original"}`
                : transcriptionSource === "enhanced"
                ? "Transcribe Enhanced · Czech"
                : "Transcribe Original · Czech"}
            </button>
          </div>

          <audio ref={audioRef} src={audioFile.url} onEnded={handleAudioEnded} className="hidden" />
        </div>
      )}

      {/* Pre-Flight Quality Check card */}
      {qualityState.status !== "idle" && (
        <div className="rounded-2xl bg-gray-900 border border-gray-800 overflow-hidden">
          <div className="px-5 py-4">

            {/* Good */}
            {qualityState.status === "good" && (
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <svg className="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="font-medium">Audio quality good</span>
                <span className="text-gray-600 text-xs ml-1 tabular-nums">
                  SNR {qualityState.snrDb.toFixed(0)} dB · {qualityState.rmsDb.toFixed(0)} dBFS
                </span>
              </div>
            )}

            {/* Warning */}
            {qualityState.status === "warn" && (
              <div className="space-y-3">
                <div className="flex items-start gap-3 rounded-xl bg-yellow-500/10 border border-yellow-500/25 px-4 py-3">
                  <span className="text-yellow-400 text-base leading-none shrink-0 mt-0.5">⚠️</span>
                  <div className="space-y-1 min-w-0">
                    <p className="text-sm font-medium text-yellow-400">Quality Warning</p>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      Background noise detected
                      {qualityState.snrDb > 0 && ` (SNR ${qualityState.snrDb.toFixed(0)} dB)`}.
                      Results may be inaccurate. Would you like to Enhance Audio?
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleEnhance}
                  disabled={isBusy}
                  className="flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium
                    bg-yellow-500/15 hover:bg-yellow-500/25 text-yellow-400
                    border border-yellow-500/30 transition-colors
                    disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
                  </svg>
                  Enhance Audio
                </button>
              </div>
            )}

            {/* Enhancing spinner */}
            {qualityState.status === "enhancing" && (
              <div className="space-y-2.5">
                {/* Top row: label + ETA + percent */}
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-blue-400">
                    <span className="w-3.5 h-3.5 rounded-full border-2 border-blue-400 border-t-transparent animate-spin shrink-0" />
                    <span>✨ Enhancing Audio…</span>
                    {/* Segment badge — visible once we know the total */}
                    {qualityState.totalSegments > 1 && (
                      <span className="text-xs px-2 py-0.5 rounded-full font-medium
                        bg-blue-500/15 text-blue-400 border border-blue-500/30 tabular-nums">
                        {qualityState.segmentIndex > 0 ? qualityState.segmentIndex : 1}
                        {" / "}
                        {qualityState.totalSegments}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-600 text-xs tabular-nums">
                      {qualityState.timeRemaining === null
                        ? qualityState.progress > 0 ? "Calculating…" : ""
                        : `${formatTime(qualityState.timeRemaining)} remaining`}
                    </span>
                    <span className="text-gray-500 tabular-nums text-sm">{qualityState.progress}%</span>
                  </div>
                </div>

                {/* Progress bar — blue→violet gradient */}
                <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${qualityState.progress}%`,
                      background: "linear-gradient(90deg, #3b82f6, #8b5cf6)",
                    }}
                  />
                </div>

                {/* Bottom row: processed time + segment detail */}
                <div className="flex items-center justify-between text-xs text-gray-500">
                  {qualityState.totalSeconds > 0 ? (
                    <span>
                      Processed:{" "}
                      <span className="text-blue-400 font-medium tabular-nums">
                        {(qualityState.processedSeconds / 60).toFixed(1)} min
                      </span>
                      {" / "}
                      <span className="tabular-nums">
                        {(qualityState.totalSeconds / 60).toFixed(1)} min
                      </span>
                    </span>
                  ) : (
                    <span>Initialising…</span>
                  )}
                  {qualityState.totalSegments > 1 ? (
                    <span className="tabular-nums text-gray-600">
                      segment{" "}
                      {qualityState.segmentIndex > 0 ? qualityState.segmentIndex : 1}
                      {" of "}
                      {qualityState.totalSegments}
                    </span>
                  ) : (
                    <span className="text-gray-600">noise reduction</span>
                  )}
                </div>
              </div>
            )}

            {/* Enhanced — Processing Chain + Verdict + Source Toggle */}
            {qualityState.status === "enhanced" && originalQuality && enhancedQuality && (() => {
              const snrDelta      = enhancedQuality.snrDb - originalQuality.snrDb;
              const isDegraded    = snrDelta < 0;
              const isReady       = !isDegraded && enhancedQuality.snrDb > 18;
              const resultLabel   = isDegraded
                ? "Reduced Clarity"
                : enhancedQuality.snrDb > 18 ? "High Quality"
                : enhancedQuality.snrDb > 10 ? "Med Quality"
                : "Low Quality";

              return (
                <div className="space-y-3">

                  {/* ── Processing Chain ──────────────────────────────────── */}
                  <div className="flex items-center gap-1.5 text-xs">
                    {/* Node: Original */}
                    <div className="flex-1 min-w-0 rounded-lg px-2.5 py-2 bg-gray-800 border border-gray-700 text-center">
                      <p className="text-gray-500 mb-0.5">Original</p>
                      <p className="text-gray-200 font-semibold tabular-nums">
                        {originalQuality.snrDb.toFixed(0)} dB
                      </p>
                    </div>

                    {/* Arrow */}
                    <svg className="w-3.5 h-3.5 text-gray-600 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                    </svg>

                    {/* Node: Enhanced */}
                    <div className={`flex-1 min-w-0 rounded-lg px-2.5 py-2 border text-center ${
                      isDegraded
                        ? "bg-yellow-500/10 border-yellow-500/30"
                        : "bg-violet-500/10 border-violet-500/30"
                    }`}>
                      <p className={`mb-0.5 ${isDegraded ? "text-yellow-500" : "text-violet-400"}`}>
                        Enhanced
                      </p>
                      <p className={`font-semibold tabular-nums ${isDegraded ? "text-yellow-300" : "text-white"}`}>
                        {enhancedQuality.snrDb.toFixed(0)} dB
                        <span className={`ml-1 font-normal ${isDegraded ? "text-red-400" : "text-emerald-400"}`}>
                          {isDegraded ? "" : "+"}{snrDelta.toFixed(0)}
                        </span>
                      </p>
                    </div>

                    {/* Arrow */}
                    <svg className="w-3.5 h-3.5 text-gray-600 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                    </svg>

                    {/* Node: Result */}
                    <div className={`flex-1 min-w-0 rounded-lg px-2.5 py-2 border text-center ${
                      isReady    ? "bg-emerald-500/10 border-emerald-500/30"
                      : isDegraded ? "bg-yellow-500/10 border-yellow-500/30"
                      :             "bg-gray-800 border-gray-700"
                    }`}>
                      <p className={`mb-0.5 ${
                        isReady ? "text-emerald-400" : isDegraded ? "text-yellow-500" : "text-gray-500"
                      }`}>
                        Result
                      </p>
                      <p className={`font-semibold ${
                        isReady ? "text-emerald-300" : isDegraded ? "text-yellow-300" : "text-gray-300"
                      }`}>
                        {resultLabel}
                      </p>
                    </div>
                  </div>

                  {/* ── Verdict ───────────────────────────────────────────── */}
                  {isReady && (
                    <div className="flex items-center gap-2 rounded-xl px-3.5 py-2.5
                      bg-emerald-500/10 border border-emerald-500/30">
                      <span className="text-emerald-400 text-sm shrink-0">✅</span>
                      <p className="text-sm font-medium text-emerald-400">Ready for Transcription</p>
                    </div>
                  )}

                  {isDegraded && (
                    <div className="space-y-2">
                      <div className="flex items-start gap-3 rounded-xl px-3.5 py-2.5
                        bg-yellow-500/10 border border-yellow-500/30">
                        <span className="text-yellow-400 text-base shrink-0 mt-0.5">⚠️</span>
                        <p className="text-sm text-yellow-400 leading-snug">
                          Enhancement reduced clarity. Consider using Original.
                        </p>
                      </div>
                      <button
                        onClick={() => setTranscriptionSource("original")}
                        className="w-full flex items-center justify-center gap-2 rounded-xl px-3 py-2 text-sm
                          font-medium border transition-colors
                          bg-gray-800/60 border-gray-600 text-gray-300
                          hover:bg-gray-700 hover:border-gray-500"
                      >
                        <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 15L3 9m0 0l6-6M3 9h12a6 6 0 010 12h-3" />
                        </svg>
                        Revert to Original
                      </button>
                    </div>
                  )}

                  {/* ── Detailed metrics ──────────────────────────────────── */}
                  <div className="rounded-xl border border-gray-700/60 overflow-hidden">
                    <div className="px-4 py-3 grid grid-cols-3 gap-3 text-xs">
                      <div className="space-y-3 pt-5">
                        <p className="text-gray-500">SNR</p>
                        <p className="text-gray-500">Level</p>
                      </div>
                      <div className="space-y-1 text-center">
                        <p className="text-gray-500 font-medium mb-2">Original</p>
                        <p className="tabular-nums text-gray-300 font-medium">{originalQuality.snrDb.toFixed(1)} dB</p>
                        <p className="tabular-nums text-gray-300 font-medium">{originalQuality.rmsDb.toFixed(1)} dBFS</p>
                      </div>
                      <div className="space-y-1 text-center">
                        <p className="text-violet-400 font-medium mb-2">✨ Enhanced</p>
                        <p className={`tabular-nums font-medium ${
                          enhancedQuality.snrDb > originalQuality.snrDb ? "text-emerald-400" : "text-red-400"
                        }`}>
                          {enhancedQuality.snrDb.toFixed(1)} dB
                        </p>
                        <p className={`tabular-nums font-medium ${
                          enhancedQuality.rmsDb > originalQuality.rmsDb ? "text-emerald-400" : "text-gray-300"
                        }`}>
                          {enhancedQuality.rmsDb.toFixed(1)} dBFS
                        </p>
                      </div>
                    </div>
                    <div className="px-4 py-2 bg-gray-800/40 border-t border-gray-700/60 flex items-center justify-between">
                      <span className="text-xs text-gray-500">Clarity change</span>
                      <span className={`text-xs font-semibold tabular-nums ${
                        snrDelta > 0 ? "text-emerald-400" : "text-red-400"
                      }`}>
                        {snrDelta > 0 ? "+" : ""}{snrDelta.toFixed(1)} dB
                      </span>
                    </div>
                  </div>

                  {/* ── Source Toggle ─────────────────────────────────────── */}
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Active source for transcription
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => setTranscriptionSource("original")}
                        className={`flex items-center justify-center gap-1.5 rounded-xl px-3 py-2.5 text-sm font-medium
                          border transition-all ${
                            transcriptionSource === "original"
                              ? "bg-gray-700 border-gray-500 text-white ring-1 ring-gray-400/30"
                              : "bg-gray-800/50 border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600"
                          }`}
                      >
                        <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                        </svg>
                        Original
                        {transcriptionSource === "original" && (
                          <svg className="w-3 h-3 text-gray-300 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                        )}
                      </button>
                      <button
                        onClick={() => setTranscriptionSource("enhanced")}
                        className={`flex items-center justify-center gap-1.5 rounded-xl px-3 py-2.5 text-sm font-medium
                          border transition-all ${
                            transcriptionSource === "enhanced"
                              ? "bg-violet-600/20 border-violet-500/60 text-violet-300 ring-1 ring-violet-400/30"
                              : "bg-gray-800/50 border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600"
                          }`}
                      >
                        ✨ Enhanced
                        {transcriptionSource === "enhanced" && (
                          <svg className="w-3 h-3 text-violet-300 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                        )}
                      </button>
                    </div>
                  </div>

                </div>
              );
            })()}

          </div>
        </div>
      )}

      {/* Status / progress / result panel */}
      {transcriptState.status !== "idle" && (
        <div className="rounded-2xl bg-gray-900 border border-gray-800 overflow-hidden">

          {/* Model loading */}
          {transcriptState.status === "loading-model" && (
            <div className="px-5 py-5 space-y-3">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className="text-gray-400">Loading Whisper model…</span>
                  {transcriptState.device && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                      transcriptState.device === "webgpu"
                        ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                        : "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                    }`}>
                      {transcriptState.device === "webgpu" ? "WebGPU" : "WASM"}
                    </span>
                  )}
                </div>
                <span className="text-gray-500 tabular-nums">{transcriptState.progress}%</span>
              </div>
              {transcriptState.file && (
                <p className="text-gray-600 text-xs truncate">{transcriptState.file}</p>
              )}
              <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
                <div
                  className="h-full bg-violet-500 rounded-full transition-all duration-300"
                  style={{ width: `${transcriptState.progress}%` }}
                />
              </div>
              <p className="text-gray-600 text-xs">
                First run downloads ~600 MB — cached for next time
              </p>
            </div>
          )}

          {/* Chunk-by-chunk transcription progress */}
          {transcriptState.status === "transcribing" && (
            <div className="px-5 py-5 space-y-3">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2 text-gray-400">
                  <span className="w-3.5 h-3.5 rounded-full border-2 border-violet-500 border-t-transparent animate-spin shrink-0" />
                  {transcriptState.statusLabel || "Transcribing…"}
                </div>
                <div className="flex items-center gap-2">
                  {/* ETA */}
                  <span className="text-gray-600 text-xs tabular-nums">
                    {transcriptState.timeRemaining === null
                      ? transcriptState.progress > 0 ? "Calculating…" : ""
                      : `${formatTime(transcriptState.timeRemaining)} remaining`}
                  </span>
                  {transcriptState.device && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                      transcriptState.device === "webgpu"
                        ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                        : "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                    }`}>
                      {transcriptState.device === "webgpu" ? "WebGPU" : "WASM"}
                    </span>
                  )}
                  <span className="text-gray-500 tabular-nums">{transcriptState.progress}%</span>
                </div>
              </div>

              {/* Progress bar */}
              <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
                <div
                  className="h-full bg-violet-500 rounded-full transition-all duration-500"
                  style={{ width: `${transcriptState.progress}%` }}
                />
              </div>

              {/* Minutes counter */}
              {transcriptState.totalSeconds > 0 && (
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>
                    Processed:{" "}
                    <span className="text-violet-400 font-medium tabular-nums">
                      {(transcriptState.processedSeconds / 60).toFixed(1)} min
                    </span>
                    {" / "}
                    <span className="tabular-nums">
                      {(transcriptState.totalSeconds / 60).toFixed(1)} min
                    </span>
                  </span>
                  <span className="tabular-nums text-gray-600">
                    chunk {transcriptState.current}/{transcriptState.total}
                  </span>
                </div>
              )}

              {/* Live partial text */}
              {transcriptState.partialText && (
                <p className="text-gray-400 text-xs leading-relaxed line-clamp-3">
                  {transcriptState.partialText}
                  <span className="inline-block w-1.5 h-3 ml-0.5 bg-violet-400 animate-pulse rounded-sm align-middle" />
                </p>
              )}
            </div>
          )}

          {/* Result */}
          {transcriptState.status === "done" && (
            <div className="divide-y divide-gray-800">

              {/* ── Transcript text ── */}
              <div className="px-5 py-5 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Transcript
                  </p>
                  <button
                    onClick={() => navigator.clipboard.writeText(transcriptState.text)}
                    className="text-xs text-gray-500 hover:text-gray-300 transition-colors flex items-center gap-1"
                    title="Copy to clipboard"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 01-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 011.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 00-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 01-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375 3.375 0 00-3.375-3.375H9.75" />
                    </svg>
                    Copy
                  </button>
                </div>
                <p className="text-white text-sm leading-relaxed whitespace-pre-wrap">
                  {transcriptState.text || (
                    <span className="text-gray-500 italic">No speech detected</span>
                  )}
                </p>
              </div>

              {/* ── Download section ── */}
              <div className="px-5 py-4 space-y-3">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Download
                </p>
                <div className="flex gap-3">
                  {/* TXT */}
                  <button
                    onClick={() => {
                      const date = new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "long", year: "numeric" });
                      const header = `Transcription — ${date}\nLanguage: Czech\n${"─".repeat(48)}\n\n`;
                      const blob = new Blob([header + transcriptState.text], { type: "text/plain;charset=utf-8" });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url; a.download = `${audioFile?.name.replace(/\.[^.]+$/, "") ?? "transcript"}.txt`; a.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="flex-1 flex items-center justify-center gap-2 rounded-xl px-4 py-2.5
                      bg-gray-800 hover:bg-gray-700 border border-gray-700
                      text-sm font-medium text-gray-300 hover:text-white transition-colors"
                  >
                    <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m.75 12l3 3m0 0l3-3m-3 3v-6m-1.5-9H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                    </svg>
                    Export as TXT
                  </button>

                  {/* DOCX */}
                  <ExportButtons
                    text={transcriptState.text}
                    filename={audioFile?.name.replace(/\.[^.]+$/, "") ?? "transcript"}
                  />
                </div>
              </div>

            </div>
          )}

          {/* Error */}
          {transcriptState.status === "error" && (
            <div className="flex items-start gap-3 px-5 py-5 text-sm text-red-400">
              <svg className="w-4 h-4 shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-.75-4.75a.75.75 0 001.5 0v-4.5a.75.75 0 00-1.5 0v4.5zm.75-7.5a.75.75 0 110 1.5.75.75 0 010-1.5z" clipRule="evenodd" />
              </svg>
              {transcriptState.message}
            </div>
          )}

        </div>
      )}
      {/* ── Smart Recommendation Toast ──────────────────────────────────────── */}
      {showQualityToast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 w-full max-w-md px-4">
          <div className="flex items-start gap-3 rounded-2xl px-4 py-3.5 shadow-xl
            bg-gray-900 border border-yellow-500/40 shadow-yellow-900/20">
            <span className="text-yellow-400 text-base shrink-0 mt-0.5">⚠️</span>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-yellow-400">Still detecting low quality</p>
              <p className="text-xs text-gray-400 mt-0.5 leading-relaxed">
                The enhanced audio still has a low SNR. For Czech lectures, consider switching to a
                slower, more precise Whisper model (e.g.{" "}
                <span className="font-mono text-gray-300">whisper-large-v3</span>) for better accuracy.
              </p>
            </div>
            <button
              onClick={() => setShowQualityToast(false)}
              className="text-gray-500 hover:text-gray-300 transition-colors shrink-0 mt-0.5"
              title="Dismiss"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

    </div>
  );
}
