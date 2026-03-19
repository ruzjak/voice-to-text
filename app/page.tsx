import AudioUploader from "@/components/AudioUploader";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-3xl space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-white">
            Audio Waveform Visualizer
          </h1>
          <p className="text-gray-400 text-lg">
            Upload an <span className="text-violet-400 font-medium">.mp3</span>
            ,{" "}
            <span className="text-violet-400 font-medium">.m4a</span>, or{" "}
            <span className="text-violet-400 font-medium">.wav</span> file to
            visualize its waveform
          </p>
        </div>
        <AudioUploader />
      </div>
    </main>
  );
}
