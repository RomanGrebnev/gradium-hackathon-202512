import clsx from "clsx";
import { ChatMessage } from "./chatHistory";
import { useAudioVisualizerBars } from "./useAudioVisualizerBars";
import { useEffect, useRef } from "react";

const PositionedAudioVisualizer = ({
  chatHistory,
  role,
  analyserNode,
  isConnected,
  onCircleClick,
  className,
}: {
  chatHistory: ChatMessage[];
  role: "user" | "assistant";
  analyserNode: AnalyserNode | null;
  isConnected: boolean;
  onCircleClick?: () => void;
  className?: string; // Optional className prop
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isAssistant = role === "assistant";

  useAudioVisualizerBars(canvasRef, {
    chatHistory,
    role,
    analyserNode,
    isConnected,
    showPlayButton: !!onCircleClick,
    clearCanvas: true,
  });

  // Resize the canvas to fit its parent element
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const size = Math.min(parent.clientWidth, parent.clientHeight);

    // If we don't do this `if` check, the recording ends up with flickering
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
    }
  });

  return (
    <div
      className={clsx(
        "max-w-3xl md:h-full flex items-center -mx-8 -my-8 px-4 md:px-0",
        isAssistant
          ? "md:w-full flex-row md:flex-row-reverse pt-36 md:pt-0"
          : "w-full flex-row-reverse md:flex-row md:pt-0 md:ml-0",
        // @ts-ignore
        className
      )}
    >
      <div
        className={clsx(
          isAssistant ? "w-40 md:w-72 2xl:w-96" : "w-full md:w-72 2xl:w-96"
        )}
      >
        <canvas
          ref={canvasRef}
          className={`w-full h-full ${onCircleClick ? "cursor-pointer" : ""
            }`}
          onClick={onCircleClick}
        />
      </div>
    </div>
  );
};

export default PositionedAudioVisualizer;
