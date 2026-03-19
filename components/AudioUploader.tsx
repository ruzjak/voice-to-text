"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import WaveformCanvas from "./WaveformCanvas";
import ExportButtons from "./ExportButtons";
import { resampleTo16kHz } from "@/lib/resample";

const ACCEPTED_EXTENSIONS = [".mp3", ".m4a", ".wav"];

interface AudioFile {
  file: File;
  name: string;
  size: number;
  url: string;
  duration: number;
  channelData: Float32Array;
}

type TranscriptState =
  | { status: "idle" }
  | { status: "loading-model"; file: string; progress: number; device: "webgpu" | "wasm" | null }
  | { status: "transcribing"; current: number; total: number; progress: number; partialText: string; statusLabel: string; processedSeconds: number; totalSeconds: number; timeRemaining: number | null; device: "webgpu" | "wasm" | null }
  | { status: "done"; text: string }
  | { status: "stopped" }
  | { status: "error"; message: string };

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

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export default function AudioUploader() {
  const [audioFile, setAudioFile]           = useState<AudioFile | null>(null);
  const [uploadError, setUploadError]       = useState<string | null>(null);
  const [loading, setLoading]               = useState(false);
  const [isDragging, setIsDragging]         = useState(false);
  const [isPlaying, setIsPlaying]           = useState(false);
  const [currentTime, setCurrentTime]       = useState(0);
  const [transcriptState, setTranscriptState] = useState<TranscriptState>({ status: "idle" });

  const deviceRef           = useRef<"webgpu" | "wasm" | null>(null);
  const inputRef            = useRef<HTMLInputElement>(null);
  const audioRef            = useRef<HTMLAudioElement>(null);
  const rafRef              = useRef<number | null>(null);
  const workerRef           = useRef<Worker | null>(null);
  const transcriptScrollRef = useRef<HTMLDivElement>(null);
  const userScrolledUp      = useRef(false);
  const stoppedTimerRef     = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (stoppedTimerRef.current) clearTimeout(stoppedTimerRef.current);
    };
  }, []);

  // ── Auto-scroll: follow new text unless user scrolled up ─────────────────
  useEffect(() => {
    const el = transcriptScrollRef.current;
    if (!el) return;
    if (!userScrolledUp.current) el.scrollTop = el.scrollHeight;
  }, [
    transcriptState.status === "transcribing"
      ? (transcriptState as { partialText: string }).partialText
      : transcriptState.status === "done"
      ? (transcriptState as { text: string }).text
      : null,
  ]);

  // ── Worker lifecycle ──────────────────────────────────────────────────────

  const getOrCreateWorker = useCallback((): Worker => {
    if (workerRef.current) return workerRef.current;

    const worker = new Worker("/worker.js", { type: "module" });

    worker.addEventListener("message", (e: MessageEvent<WorkerMessage>) => {
      const msg = e.data;

      if (msg.type === "status" && msg.status === "loading") {
        setTranscriptState({ status: "loading-model", file: "", progress: 0, device: deviceRef.current });

      } else if (msg.type === "device") {
        deviceRef.current = msg.device;
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

  // ── Transcription ─────────────────────────────────────────────────────────

  const handleTranscribe = useCallback(async () => {
    if (!audioFile) return;
    userScrolledUp.current = false;
    setTranscriptState({ status: "loading-model", file: "", progress: 0, device: deviceRef.current });

    try {
      const audio = await resampleTo16kHz(audioFile.file);
      const worker = getOrCreateWorker();
      worker.postMessage({ type: "load" });
      worker.postMessage({ type: "transcribe", audio }, [audio.buffer]);
    } catch (err) {
      setTranscriptState({
        status: "error",
        message: err instanceof Error ? err.message : "Failed to process audio",
      });
    }
  }, [audioFile, getOrCreateWorker]);

  // ── Stop ──────────────────────────────────────────────────────────────────

  const handleStop = useCallback(() => {
    if (
      transcriptState.status === "loading-model" ||
      transcriptState.status === "transcribing"
    ) {
      workerRef.current?.terminate();
      workerRef.current = null;
      setTranscriptState({ status: "stopped" });
      if (stoppedTimerRef.current) clearTimeout(stoppedTimerRef.current);
      stoppedTimerRef.current = setTimeout(() => {
        setTranscriptState((prev) => prev.status === "stopped" ? { status: "idle" } : prev);
      }, 3000);
    }
  }, [transcriptState.status]);

  // ── File processing ───────────────────────────────────────────────────────

  const processFile = useCallback(async (file: File) => {
    setUploadError(null);
    setAudioFile(null);
    setIsPlaying(false);
    setCurrentTime(0);
    setTranscriptState({ status: "idle" });

    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(`.${ext}`)) {
      setUploadError("Unsupported file type. Please upload .mp3, .m4a, or .wav files.");
      return;
    }

    setLoading(true);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioCtx    = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
      await audioCtx.close();

      const rawData   = audioBuffer.getChannelData(0);
      const samples   = 4000;
      const blockSize = Math.floor(rawData.length / samples);
      const channelData = new Float32Array(samples);
      for (let i = 0; i < samples; i++) {
        let sum = 0;
        for (let j = 0; j < blockSize; j++) sum += Math.abs(rawData[i * blockSize + j]);
        channelData[i] = sum / blockSize;
      }

      setAudioFile({
        file,
        name:        file.name,
        size:        file.size,
        url:         URL.createObjectURL(file),
        duration:    audioBuffer.duration,
        channelData,
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
    if (inputRef.current) inputRef.current.value = "";
  };

  const isBusy = transcriptState.status === "loading-model" || transcriptState.status === "transcribing";
  const waveformProgress = audioFile ? currentTime / audioFile.duration : 0;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4">

      {/* ── Drop zone ─────────────────────────────────────────────────────── */}
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

      {/* ── Upload error ──────────────────────────────────────────────────── */}
      {uploadError && (
        <div className="flex items-center gap-3 rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-400">
          <svg className="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-.75-4.75a.75.75 0 001.5 0v-4.5a.75.75 0 00-1.5 0v4.5zm.75-7.5a.75.75 0 110 1.5.75.75 0 010-1.5z" clipRule="evenodd" />
          </svg>
          {uploadError}
        </div>
      )}

      {/* ── Player card ───────────────────────────────────────────────────── */}
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
                <p className="text-white font-medium text-sm truncate">{audioFile.name}</p>
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
              className="w-full flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-medium
                transition-all active:scale-[0.99] text-white
                bg-violet-600 hover:bg-violet-500
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
                ? "Transcribe Again · Czech"
                : "Transcribe Lecture · Czech"}
            </button>
          </div>

          <audio ref={audioRef} src={audioFile.url} onEnded={handleAudioEnded} className="hidden" />
        </div>
      )}

      {/* ── Progress / result card ────────────────────────────────────────── */}
      {transcriptState.status !== "idle" && (
        <div className="rounded-2xl bg-gray-900 border border-gray-800 overflow-hidden">

          {/* Model loading */}
          {transcriptState.status === "loading-model" && (
            <div className="px-5 py-5 space-y-3">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 rounded-full border-2 border-violet-500 border-t-transparent animate-spin shrink-0" />
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
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 tabular-nums">{transcriptState.progress}%</span>
                  <button
                    onClick={handleStop}
                    className="flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium
                      border border-red-500/50 text-red-400 bg-red-500/10
                      hover:bg-red-500/20 hover:border-red-500/70 transition-colors"
                  >
                    <svg className="w-3 h-3 shrink-0" fill="currentColor" viewBox="0 0 24 24">
                      <rect x="6" y="6" width="12" height="12" rx="1" />
                    </svg>
                    Stop
                  </button>
                </div>
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

          {/* Transcribing */}
          {transcriptState.status === "transcribing" && (
            <div className="px-5 py-5 space-y-3">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2 text-gray-400">
                  <span className="w-3.5 h-3.5 rounded-full border-2 border-violet-500 border-t-transparent animate-spin shrink-0" />
                  {transcriptState.statusLabel || "Transcribing…"}
                </div>
                <div className="flex items-center gap-2">
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
                  <button
                    onClick={handleStop}
                    className="flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium
                      border border-red-500/50 text-red-400 bg-red-500/10
                      hover:bg-red-500/20 hover:border-red-500/70 transition-colors"
                  >
                    <svg className="w-3 h-3 shrink-0" fill="currentColor" viewBox="0 0 24 24">
                      <rect x="6" y="6" width="12" height="12" rx="1" />
                    </svg>
                    Stop
                  </button>
                </div>
              </div>

              <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
                <div
                  className="h-full bg-violet-500 rounded-full transition-all duration-500"
                  style={{ width: `${transcriptState.progress}%` }}
                />
              </div>

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
                <div
                  ref={transcriptScrollRef}
                  onScroll={() => {
                    const el = transcriptScrollRef.current;
                    if (!el) return;
                    userScrolledUp.current = el.scrollHeight - el.scrollTop - el.clientHeight > 32;
                  }}
                  className="transcript-scroll min-h-[200px] max-h-[600px] overflow-y-auto rounded-xl
                    bg-gray-800/40 border border-gray-700/50 px-4 py-3"
                >
                  <p className="text-gray-400 text-sm leading-relaxed whitespace-pre-wrap">
                    {transcriptState.partialText}
                    <span className="inline-block w-1.5 h-3 ml-0.5 bg-violet-400 animate-pulse rounded-sm align-middle" />
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Done */}
          {transcriptState.status === "done" && (
            <div className="divide-y divide-gray-800">

              {/* Transcript preview */}
              <div className="px-5 py-5 space-y-3">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Transcript</p>
                <div
                  ref={transcriptScrollRef}
                  onScroll={() => {
                    const el = transcriptScrollRef.current;
                    if (!el) return;
                    userScrolledUp.current = el.scrollHeight - el.scrollTop - el.clientHeight > 32;
                  }}
                  className="transcript-scroll min-h-[200px] max-h-[600px] overflow-y-auto rounded-xl
                    bg-gray-800/40 border border-gray-700/50 px-4 py-3"
                >
                  {transcriptState.text ? (
                    <p className="text-white text-sm leading-relaxed whitespace-pre-wrap">
                      {transcriptState.text}
                    </p>
                  ) : (
                    <p className="text-gray-500 italic text-sm">No speech detected</p>
                  )}
                </div>
              </div>

              {/* Export */}
              <div className="px-5 py-4 space-y-3">
                <div className="flex items-center gap-2">
                  <span className="text-emerald-400 text-sm">✅</span>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Export
                  </p>
                </div>
                <ExportButtons
                  text={transcriptState.text}
                  filename={`lecture_notes_${audioFile?.name.replace(/\.[^.]+$/, "") ?? "transcript"}`}
                  durationSeconds={audioFile?.duration}
                />
              </div>

            </div>
          )}

          {/* Stopped */}
          {transcriptState.status === "stopped" && (
            <div className="flex items-center gap-3 px-5 py-4 text-sm text-red-400">
              <svg className="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
              <span>Process terminated by user.</span>
            </div>
          )}

          {/* Error */}
          {transcriptState.status === "error" && (
            <div className="flex items-start gap-3 px-5 py-5 text-sm text-red-400">
              <svg className="w-4 h-4 shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-.75-4.75a.75.75 0 001.5 0v-4.5a.75.75 0 00-1.5 0v4.5zm.75-7.5a.75.75 0 110 1.5.75.75 0 010-1.5z" clipRule="evenodd" />
              </svg>
              <span>{transcriptState.message}</span>
            </div>
          )}

        </div>
      )}

    </div>
  );
}
