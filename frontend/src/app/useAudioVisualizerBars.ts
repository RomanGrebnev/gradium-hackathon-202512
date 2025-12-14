import { useRef, useEffect, useState } from "react";
import { getCSSVariable } from "./cssUtil";
import { ChatMessage } from "./chatHistory";

const BAR_COUNT = 5;
const BAR_WIDTH_RATIO = 0.16; // Width relative to total width
const BAR_GAP_RATIO = 0.02; // Gap between bars relative to total width
const MIN_BAR_HEIGHT_RATIO = 0.1; // Minimum bar height as ratio of max height
const SMOOTHING_FACTOR = 0.7; // Smoothing for bar animation

const WIDTH_INACTIVE = 3;
const WIDTH_ACTIVE = 5;

// Size scale factors for connected/disconnected states
const SCALE_CONNECTED = 1;
const SCALE_DISCONNECTED = 0.8;
const ANIMATION_DURATION = 500; // milliseconds

const INTERRUPTION_CHAR = "—"; // em-dash

const getAnalyzerData = (
  analyserNode: AnalyserNode | null
): Float32Array => {
  const fftSize = 2048;
  const frequencyData = new Float32Array(fftSize / 2);

  if (!analyserNode) {
    // return arrays corresponding to silence
    frequencyData.fill(-100); // -100 dBFS
    return frequencyData;
  } else {
    // Configure analyzer node
    analyserNode.fftSize = fftSize;
    analyserNode.smoothingTimeConstant = 0.85;

    analyserNode.getFloatFrequencyData(frequencyData);

    return frequencyData;
  }
};

const getIsActive = (
  chatHistory: ChatMessage[],
  role: "user" | "assistant"
) => {
  // Find the latest non-empty message from the specified role
  for (let i = chatHistory.length - 1; i >= 0; i--) {
    const message = chatHistory[i];

    // Empty messages, or ones where the LLM started generating but was interrupted
    // before it said anything
    if (message.content === "" || message.content === INTERRUPTION_CHAR)
      continue;

    if (message.content === "...") {
      // The user is silent, no more speech is coming
      return false;
    }

    if (message.role === role) {
      return true;
    } else {
      return false;
    }
  }
  // No non-empty messages found
  return false;
};

// Function to draw bars visualization
const drawBarsVisualization = (
  canvas: HTMLCanvasElement,
  canvasCtx: CanvasRenderingContext2D,
  barHeights: number[],
  colorName: string,
  lineWidth: number,
  animationProgress: number
) => {
  const width = canvas.width;
  const height = canvas.height;

  // Calculate scale factor from animation progress (0 = disconnected, 1 = connected)
  const scaleFactor =
    SCALE_DISCONNECTED +
    (SCALE_CONNECTED - SCALE_DISCONNECTED) * animationProgress;

  const barWidth = width * BAR_WIDTH_RATIO;
  const barGap = width * BAR_GAP_RATIO;
  const totalBarsWidth = BAR_COUNT * barWidth + (BAR_COUNT - 1) * barGap;
  const startX = (width - totalBarsWidth) / 2;
  const maxBarHeight = height * 0.8; // Use 80% of canvas height for bars

  canvasCtx.fillStyle = getCSSVariable(colorName);

  for (let i = 0; i < BAR_COUNT; i++) {
    const x = startX + i * (barWidth + barGap);
    const barHeight =
      (barHeights[i] * maxBarHeight + MIN_BAR_HEIGHT_RATIO * maxBarHeight) *
      scaleFactor;

    const y = height / 2 - barHeight / 2;

    // Draw filled rectangle with rounded corners
    const radius = barWidth / 2; // Fully rounded ends (semi-circles)
    canvasCtx.beginPath();
    canvasCtx.roundRect(x, y, barWidth, barHeight, radius);
    canvasCtx.fill();
  }
};

// New function to draw a play triangle
const drawPlayButton = (
  canvas: HTMLCanvasElement,
  canvasCtx: CanvasRenderingContext2D,
  colorName: string,
  animationProgress: number
) => {
  // Calculate opacity based on animation progress (0 = fully visible, 1 = invisible)
  const opacity = 1 - animationProgress;

  if (opacity <= 0) return; // Don't draw if fully transparent

  const centerX = canvas.width / 2;
  const centerY = canvas.height * 0.65; // Position below center (below bars)
  const size = Math.min(canvas.width, canvas.height) * 0.15; // Play button size

  // Create triangle path
  canvasCtx.beginPath();
  canvasCtx.moveTo(centerX + size / 2, centerY);
  canvasCtx.lineTo(centerX - size / 4, centerY - size / 2);
  canvasCtx.lineTo(centerX - size / 4, centerY + size / 2);
  canvasCtx.closePath();

  // Fill with color and opacity
  const color = getCSSVariable(colorName);
  // Parse the CSS variable color to get RGB values
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");
  if (!tempCtx) return;

  tempCtx.fillStyle = color;
  tempCtx.fillRect(0, 0, 1, 1);
  const rgba = tempCtx.getImageData(0, 0, 1, 1).data;

  // Apply opacity to the color
  canvasCtx.fillStyle = `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${opacity})`;
  canvasCtx.fill();

  // Add white border
  canvasCtx.strokeStyle = `rgba(140, 140, 140, ${opacity})`;
  canvasCtx.lineWidth = 3;
  canvasCtx.stroke();
};

export interface UseAudioVisualizerBarsOptions {
  chatHistory: ChatMessage[];
  role: "user" | "assistant";
  analyserNode: AnalyserNode | null;
  isConnected?: boolean;
  showPlayButton?: boolean;
  clearCanvas: boolean;
}

export const useAudioVisualizerBars = (
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  options: UseAudioVisualizerBarsOptions
) => {
  const {
    chatHistory,
    role,
    analyserNode,
    isConnected = false,
    showPlayButton = false,
    clearCanvas,
  } = options;

  const isActive = getIsActive(chatHistory, role);
  const isAssistant = role === "assistant";
  const colorName = isAssistant ? "color-blue" : "color-black";

  const animationRef = useRef<number>(-1);
  const barHeights = useRef<number[]>(new Array(BAR_COUNT).fill(0));
  const targetBarHeights = useRef<number[]>(new Array(BAR_COUNT).fill(0));
  const animationFrameRef = useRef<number | null>(null);
  const animationStartTime = useRef<number | null>(null);
  const animationPreviousProgress = useRef<number>(isConnected ? 1 : 0);

  const interruptionTimeRef = useRef(0);
  const [interruptionIndex, setInterruptionIndex] = useState(0);

  // Single state for animation progress: 0 = disconnected state, 1 = connected state
  const [animationProgress, setAnimationProgress] = useState<number>(
    isConnected ? 1 : 0
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const stream = canvas.captureStream(30);
    stream.getTracks().forEach((track) => {
      track.stop();
    });
  }, [canvasRef]);

  useEffect(() => {
    if (chatHistory.length > interruptionIndex) {
      if (
        role === "user" &&
        chatHistory[chatHistory.length - 1].role === "assistant" &&
        // An interruption
        chatHistory[chatHistory.length - 1].content.endsWith(
          INTERRUPTION_CHAR
        ) &&
        // but not *only* an interruption char. That would mean the LLM got interrupted
        // before it said anything, and we don't want to count that as an interruption
        chatHistory[chatHistory.length - 1].content !== INTERRUPTION_CHAR
      ) {
        interruptionTimeRef.current = Date.now();
        setInterruptionIndex(chatHistory.length);
      }
    }
  }, [chatHistory, interruptionIndex, role]);

  // Handle connection state changes
  useEffect(() => {
    const animate = (timestamp: number) => {
      if (!animationStartTime.current) {
        animationStartTime.current = timestamp;
        animationPreviousProgress.current = animationProgress;
      }

      const elapsed = timestamp - animationStartTime.current;
      const duration = ANIMATION_DURATION;
      const progress = Math.min(elapsed / duration, 1);

      const targetProgress = isConnected ? 1 : 0;

      // Ease in-out function for smoother animation
      const easeInOutCubic = (t: number): number => {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      };

      const newProgress =
        animationPreviousProgress.current +
        easeInOutCubic(progress) *
          (targetProgress - animationPreviousProgress.current);

      setAnimationProgress(newProgress);

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        // Animation complete
        animationFrameRef.current = null;
        animationStartTime.current = null;
        setAnimationProgress(targetProgress); // Ensure we end exactly at target value
      }
    };

    // Cancel any existing animation
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Start the animation
    animationFrameRef.current = requestAnimationFrame(animate);

    // Cleanup on unmount
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isConnected, animationProgress]);

  // Main drawing effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const canvasCtx = canvas.getContext("2d");
    if (!canvasCtx) return;

    const draw = () => {
      const frequencyData = getAnalyzerData(analyserNode);

      // Use logarithmic frequency distribution with more aggressive ranges
      // to ensure bars 3 and 4 are properly utilized
      const frequencyRanges = [
        [0, 0.03],      // Bar 0: Very low frequencies (0-3%)
        [0.03, 0.08],   // Bar 1: Low frequencies (3-8%)
        [0.08, 0.18],   // Bar 2: Mid frequencies (8-18%)
        [0.18, 0.40],   // Bar 3: Mid-high frequencies (18-40%)
        [0.40, 1.0],    // Bar 4: High frequencies (40-100%)
      ];

      for (let i = 0; i < BAR_COUNT; i++) {
        const [startRatio, endRatio] = frequencyRanges[i];
        const startBin = Math.floor(startRatio * frequencyData.length);
        const endBin = Math.floor(endRatio * frequencyData.length);

        let sum = 0;
        for (let j = startBin; j < endBin; j++) {
          sum += frequencyData[j];
        }
        const avgDbfs = sum / (endBin - startBin);

        // Normalize from dBFS (-100 to 0) to 0-1 range
        const normalized = Math.max(0, Math.min(1, (avgDbfs + 100) / 100));

        // Apply some scaling to make it more visually interesting
        // Use much lower exponents for bars 3 and 4 to boost their visibility
        const exponent = 0.5 - i * 0.08; // More aggressive decrease for higher bars
        targetBarHeights.current[i] = Math.pow(normalized, exponent);
      }

      // Smooth the bar heights
      for (let i = 0; i < BAR_COUNT; i++) {
        barHeights.current[i] +=
          (targetBarHeights.current[i] - barHeights.current[i]) *
          (1 - SMOOTHING_FACTOR);
      }

      // Schedule the next animation frame
      animationRef.current = requestAnimationFrame(draw);

      if (clearCanvas) {
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      }

      const secSinceInterruption =
        (Date.now() - interruptionTimeRef.current) / 1000;
      const widthScale =
        1 + 2 * Math.exp(-Math.pow(secSinceInterruption * 3, 2));

      drawBarsVisualization(
        canvas,
        canvasCtx,
        barHeights.current,
        colorName,
        Math.max(
          isActive ? WIDTH_ACTIVE : WIDTH_INACTIVE,
          WIDTH_INACTIVE * widthScale
        ),
        animationProgress
      );

      // Draw play button if we have onClick and not fully connected
      if (showPlayButton && animationProgress < 1) {
        drawPlayButton(canvas, canvasCtx, colorName, animationProgress);
      }
    };

    // Start the animation
    draw();

    // Clean up the animation on unmount
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [
    analyserNode,
    colorName,
    isActive,
    animationProgress,
    showPlayButton,
    canvasRef,
    clearCanvas,
  ]);

  return {};
};
