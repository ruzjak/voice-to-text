import { pipeline, env } from "@xenova/transformers";

// Always fetch models from the Hugging Face CDN
env.allowLocalModels = false;

type IncomingMessage =
  | { type: "load" }
  | { type: "transcribe"; audio: Float32Array };

export type WorkerMessage =
  | { type: "loading" }
  | { type: "progress"; file: string; progress: number }
  | { type: "ready" }
  | { type: "transcribing" }
  | { type: "result"; text: string }
  | { type: "error"; message: string };

let transcriber: Awaited<ReturnType<typeof pipeline>> | null = null;

async function loadModel() {
  self.postMessage({ type: "loading" } satisfies WorkerMessage);

  transcriber = await pipeline(
    "automatic-speech-recognition",
    "Xenova/whisper-small",
    {
      progress_callback: (p: {
        status: string;
        file?: string;
        progress?: number;
      }) => {
        if (p.status === "progress" && p.file !== undefined) {
          self.postMessage({
            type: "progress",
            file: p.file,
            progress: Math.round(p.progress ?? 0),
          } satisfies WorkerMessage);
        }
      },
    }
  );

  self.postMessage({ type: "ready" } satisfies WorkerMessage);
}

async function transcribe(audio: Float32Array) {
  if (!transcriber) throw new Error("Model not loaded");

  self.postMessage({ type: "transcribing" } satisfies WorkerMessage);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const output = await (transcriber as any)(audio as any, {
    task: "transcribe",
    language: "cs",
    return_timestamps: false,
  });

  const result = output as { text: string } | Array<{ text: string }>;
  const text = Array.isArray(result) ? result[0].text : result.text;

  self.postMessage({ type: "result", text: text.trim() } satisfies WorkerMessage);
}

self.addEventListener("message", async (e: MessageEvent<IncomingMessage>) => {
  try {
    if (e.data.type === "load") {
      await loadModel();
    } else if (e.data.type === "transcribe") {
      await transcribe(e.data.audio);
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    } satisfies WorkerMessage);
  }
});
