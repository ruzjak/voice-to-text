"use client";

import { useCallback, useEffect, useRef } from "react";

interface WaveformCanvasProps {
  channelData: Float32Array;
  progress: number; // 0–1
  onSeek: (ratio: number) => void;
}

const BAR_WIDTH = 2;
const BAR_GAP = 1;
const BAR_RADIUS = 1;
const HEIGHT = 120;

const COLOR_PLAYED = "#8b5cf6";   // violet-500
const COLOR_REMAINING = "#374151"; // gray-700

export default function WaveformCanvas({
  channelData,
  progress,
  onSeek,
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth;
    canvas.width = width * dpr;
    canvas.height = HEIGHT * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${HEIGHT}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, HEIGHT);

    const totalBars = Math.floor(width / (BAR_WIDTH + BAR_GAP));
    const step = Math.floor(channelData.length / totalBars);
    const mid = HEIGHT / 2;

    // Normalize amplitude
    let max = 0;
    for (let i = 0; i < channelData.length; i++) {
      if (channelData[i] > max) max = channelData[i];
    }
    const norm = max > 0 ? 1 / max : 1;

    const playedX = progress * width;

    for (let i = 0; i < totalBars; i++) {
      let sum = 0;
      for (let j = 0; j < step; j++) {
        sum += channelData[i * step + j] || 0;
      }
      const amplitude = (sum / step) * norm;
      const barHeight = Math.max(2, amplitude * (HEIGHT * 0.88));
      const x = i * (BAR_WIDTH + BAR_GAP);
      const y = mid - barHeight / 2;

      ctx.fillStyle = x < playedX ? COLOR_PLAYED : COLOR_REMAINING;

      // Rounded rectangle
      ctx.beginPath();
      ctx.roundRect(x, y, BAR_WIDTH, barHeight, BAR_RADIUS);
      ctx.fill();
    }

    // Playhead line
    if (progress > 0 && progress < 1) {
      ctx.beginPath();
      ctx.moveTo(playedX, 0);
      ctx.lineTo(playedX, HEIGHT);
      ctx.strokeStyle = "#c4b5fd"; // violet-300
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [channelData, progress]);

  // Redraw when data or progress changes
  useEffect(() => {
    draw();
  }, [draw]);

  // Redraw on resize
  useEffect(() => {
    const observer = new ResizeObserver(() => draw());
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [draw]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    onSeek(Math.max(0, Math.min(1, ratio)));
  };

  return (
    <div ref={containerRef} className="w-full">
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        className="w-full cursor-pointer"
        style={{ height: HEIGHT }}
      />
    </div>
  );
}
