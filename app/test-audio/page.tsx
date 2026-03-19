"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Status = "pending" | "running" | "ready" | "error";

interface Check {
  label: string;
  status: Status;
  detail: string;
  progress?: number; // 0-100, only during "running" for model load
}

const INITIAL_CHECKS: Check[] = [
  { label: "AudioContext", status: "pending", detail: "Not checked yet" },
  { label: "COOP / COEP headers (WASM isolation)", status: "pending", detail: "Not checked yet" },
  { label: "Whisper-tiny pipeline init", status: "pending", detail: "Not checked yet" },
  { label: "Dry Run (silent 1 s buffer)", status: "pending", detail: "Not triggered yet" },
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Pipeline = (audio: Float32Array, opts: Record<string, unknown>) => Promise<any>;

export default function TestAudioPage() {
  const [checks, setChecks] = useState<Check[]>(INITIAL_CHECKS);
  const [dryRunLog, setDryRunLog] = useState<string | null>(null);
  const pipelineRef = useRef<Pipeline | null>(null);
  const runningRef = useRef(false);

  const patch = useCallback((index: number, update: Partial<Check>) => {
    setChecks((prev) => prev.map((c, i) => (i === index ? { ...c, ...update } : c)));
  }, []);

  const runChecks = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    pipelineRef.current = null;
    setDryRunLog(null);
    setChecks(INITIAL_CHECKS);

    // ── Check 1: AudioContext ───────────────────────────────────────────────
    patch(0, { status: "running", detail: "Checking…" });
    await tick();
    try {
      console.log("[check-1] new AudioContext()");
      const ctx = new AudioContext();
      await ctx.close();
      console.log("[check-1] AudioContext state after close:", ctx.state);
      patch(0, { status: "ready", detail: `Available — state: ${ctx.state}` });
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      console.error("[check-1] FAILED", err.stack);
      patch(0, { status: "error", detail: err.message });
    }

    // ── Check 2: COOP / COEP ────────────────────────────────────────────────
    patch(1, { status: "running", detail: "Checking headers…" });
    await tick();
    try {
      // crossOriginIsolated is true only when COOP+COEP headers are both present
      const isolated = window.crossOriginIsolated;
      console.log("[check-2] crossOriginIsolated:", isolated);
      // SharedArrayBuffer construction throws if isolation is missing
      new SharedArrayBuffer(1);
      console.log("[check-2] SharedArrayBuffer OK");
      patch(1, {
        status: "ready",
        detail: `crossOriginIsolated=${isolated} — SharedArrayBuffer OK`,
      });
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      console.error("[check-2] FAILED", err.stack);
      patch(1, {
        status: "error",
        detail: `crossOriginIsolated=${window.crossOriginIsolated} — ${err.message}`,
      });
    }

    // ── Check 3: Pipeline init ──────────────────────────────────────────────
    patch(2, { status: "running", detail: "Importing @xenova/transformers…", progress: 0 });
    await tick();
    try {
      console.group("[check-3] Whisper-tiny pipeline init");

      console.log("[check-3] dynamic import @xenova/transformers");
      const { pipeline, env } = await import("@xenova/transformers");

      console.log("[check-3] configuring env");
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      console.log("[check-3] env.allowLocalModels =", env.allowLocalModels);
      console.log("[check-3] env.useBrowserCache  =", env.useBrowserCache);

      console.log("[check-3] calling pipeline()");
      const pipe = await pipeline(
        "automatic-speech-recognition",
        "Xenova/whisper-tiny",
        {
          quantized: true,
          progress_callback: (p: { status: string; file?: string; progress?: number }) => {
            if (p.status === "progress") {
              const pct = Math.round(p.progress ?? 0);
              console.log(`[check-3] downloading ${p.file} — ${pct}%`);
              patch(2, {
                status: "running",
                detail: p.file ?? "Downloading…",
                progress: pct,
              });
            }
          },
        }
      );

      console.log("[check-3] pipeline ready:", pipe);
      console.groupEnd();

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      pipelineRef.current = pipe as any;
      patch(2, { status: "ready", detail: "Xenova/whisper-tiny loaded (quantized)", progress: undefined });
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      console.error("[check-3] FAILED", err);
      console.error("[check-3] stack:", err.stack);
      console.groupEnd();
      patch(2, {
        status: "error",
        detail: `${err.message} — see console for stack trace`,
        progress: undefined,
      });
    }

    runningRef.current = false;
  }, [patch]);

  // Auto-run on mount
  useEffect(() => { runChecks(); }, [runChecks]);

  const handleDryRun = async () => {
    if (!pipelineRef.current) return;
    patch(3, { status: "running", detail: "Generating silent buffer…" });
    setDryRunLog(null);
    await tick();

    try {
      console.group("[dry-run]");
      // 1 second of silence at 16 kHz
      const silent = new Float32Array(16_000);
      console.log("[dry-run] silent buffer length:", silent.length);
      patch(3, { status: "running", detail: "Running inference…" });
      await tick();

      console.log("[dry-run] calling pipeline with silent buffer");
      const result = await pipelineRef.current(silent, {
        task: "transcribe",
        language: "cs",
        return_timestamps: false,
      });

      const log = JSON.stringify(result, null, 2);
      console.log("[dry-run] result:", result);
      console.groupEnd();
      setDryRunLog(log);
      patch(3, {
        status: "ready",
        detail: `Inference succeeded — text: "${result?.text?.trim() || "(empty)"}"`,
      });
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      console.error("[dry-run] FAILED", err.stack);
      console.groupEnd();
      patch(3, { status: "error", detail: `${err.message} — see console for stack trace` });
    }
  };

  const modelReady = checks[2].status === "ready";
  const dryRunRunning = checks[3].status === "running";

  return (
    <main className="min-h-screen bg-gray-950 text-white px-4 py-16">
      <div className="max-w-2xl mx-auto space-y-8">

        {/* Header */}
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-yellow-500/10 text-yellow-400 border border-yellow-500/20">
              /test-audio
            </span>
            <span className="text-xs text-gray-600">temporary diagnostics page</span>
          </div>
          <h1 className="text-2xl font-bold">Audio Pipeline Diagnostics</h1>
          <p className="text-gray-400 text-sm">
            Verifies browser capabilities and the Whisper-tiny pipeline end-to-end.
          </p>
        </div>

        {/* Check cards */}
        <div className="space-y-3">
          {checks.map((check, i) => (
            <CheckCard key={check.label} check={check} index={i} />
          ))}
        </div>

        {/* Actions */}
        <div className="flex gap-3 flex-wrap">
          <button
            onClick={runChecks}
            disabled={runningRef.current}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
              bg-gray-800 hover:bg-gray-700 border border-gray-700 transition-colors
              disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
            </svg>
            Re-run checks
          </button>

          <button
            onClick={handleDryRun}
            disabled={!modelReady || dryRunRunning}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
              bg-violet-600 hover:bg-violet-500 transition-colors
              disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {dryRunRunning ? (
              <span className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
              </svg>
            )}
            Dry Run
          </button>
        </div>

        {/* Dry-run log */}
        {dryRunLog && (
          <div className="rounded-xl bg-gray-900 border border-gray-800 overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-800">
              <span className="text-xs font-medium text-gray-400">Pipeline output</span>
              <button
                onClick={() => navigator.clipboard.writeText(dryRunLog)}
                className="text-xs text-gray-600 hover:text-gray-300 transition-colors"
              >
                Copy
              </button>
            </div>
            <pre className="px-4 py-3 text-xs text-green-400 overflow-x-auto leading-relaxed">
              {dryRunLog}
            </pre>
          </div>
        )}

        {/* WASM header note */}
        <div className="rounded-xl bg-blue-500/5 border border-blue-500/20 px-4 py-3 text-xs text-blue-300 space-y-1">
          <p className="font-medium">WASM / COOP-COEP note</p>
          <p className="text-blue-400/70">
            Headers <code className="text-blue-300">Cross-Origin-Opener-Policy: same-origin</code> and{" "}
            <code className="text-blue-300">Cross-Origin-Embedder-Policy: require-corp</code> are injected by{" "}
            <code className="text-blue-300">next.config.ts</code> for all routes. These enable{" "}
            <code className="text-blue-300">SharedArrayBuffer</code> which the ONNX WASM backend
            requires for multi-threaded execution.
          </p>
        </div>

      </div>
    </main>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function CheckCard({ check, index }: { check: Check; index: number }) {
  return (
    <div className="flex items-start gap-4 rounded-xl bg-gray-900 border border-gray-800 px-4 py-4">
      <div className="mt-0.5 shrink-0">
        <StatusIcon status={check.status} />
      </div>
      <div className="flex-1 min-w-0 space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-600 tabular-nums">#{index + 1}</span>
          <span className="text-sm font-medium text-white">{check.label}</span>
          <StatusBadge status={check.status} />
        </div>
        <p className="text-xs text-gray-400 truncate" title={check.detail}>
          {check.detail}
        </p>
        {check.status === "running" && check.progress !== undefined && (
          <div className="space-y-1">
            <div className="h-1 rounded-full bg-gray-800 overflow-hidden">
              <div
                className="h-full bg-violet-500 rounded-full transition-all duration-200"
                style={{ width: `${check.progress}%` }}
              />
            </div>
            <span className="text-xs text-gray-600 tabular-nums">{check.progress}%</span>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusIcon({ status }: { status: Status }) {
  if (status === "pending")
    return <div className="w-5 h-5 rounded-full border-2 border-gray-700" />;
  if (status === "running")
    return (
      <div className="w-5 h-5 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
    );
  if (status === "ready")
    return (
      <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
      </svg>
    );
  return (
    <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

function StatusBadge({ status }: { status: Status }) {
  const styles: Record<Status, string> = {
    pending: "bg-gray-800 text-gray-500 border-gray-700",
    running: "bg-violet-500/10 text-violet-400 border-violet-500/30",
    ready:   "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    error:   "bg-red-500/10 text-red-400 border-red-500/30",
  };
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded border font-medium ${styles[status]}`}>
      {status === "pending" ? "pending"
        : status === "running" ? "running…"
        : status === "ready" ? "Ready"
        : "Error"}
    </span>
  );
}

function tick() {
  return new Promise<void>((r) => setTimeout(r, 30));
}
