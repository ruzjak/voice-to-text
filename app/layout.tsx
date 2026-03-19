import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice to Text – Audio Waveform Visualizer",
  description: "Upload an audio file and visualize its waveform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-white min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
