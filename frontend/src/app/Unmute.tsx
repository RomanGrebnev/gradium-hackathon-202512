"use client";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { useCallback, useEffect, useState } from "react";
import { useMicrophoneAccess } from "./useMicrophoneAccess";
import { base64DecodeOpus, base64EncodeOpus } from "./audioUtil";
import SlantedButton from "@/app/SlantedButton";
import { useAudioProcessor as useAudioProcessor } from "./useAudioProcessor";
import useKeyboardShortcuts from "./useKeyboardShortcuts";
import { prettyPrintJson } from "pretty-print-json";
import PositionedAudioVisualizer from "./PositionedAudioVisualizer";
import ConfigPage from "./ConfigPage";
import {
  DEFAULT_UNMUTE_CONFIG,
  UnmuteConfig,
} from "./UnmuteConfigurator";
import { HealthStatus } from "./CouldNotConnect";
import Subtitles from "./Subtitles";
import { ChatMessage, compressChatHistory } from "./chatHistory";
import useWakeLock from "./useWakeLock";
import ErrorMessages, { ErrorItem, makeErrorItem } from "./ErrorMessages";
import { useRecordingCanvas } from "./useRecordingCanvas";
import { useGoogleAnalytics } from "./useGoogleAnalytics";
import clsx from "clsx";
import { useBackendServerUrl } from "./useBackendServerUrl";
import { RECORDING_CONSENT_STORAGE_KEY } from "./ConsentModal";
import Transcript from "./Transcript";
import ReportPage from "./ReportPage";

const Unmute = () => {
  const { isDevMode, showSubtitles } = useKeyboardShortcuts();
  const [debugDict, setDebugDict] = useState<object | null>(null);
  const [unmuteConfig, setUnmuteConfig] = useState<UnmuteConfig>(
    DEFAULT_UNMUTE_CONFIG
  );
  const [rawChatHistory, setRawChatHistory] = useState<ChatMessage[]>([]);
  const chatHistory = compressChatHistory(rawChatHistory);

  const { microphoneAccess, askMicrophoneAccess } = useMicrophoneAccess();

  const [shouldConnect, setShouldConnect] = useState(false);
  const backendServerUrl = useBackendServerUrl();
  const [webSocketUrl, setWebSocketUrl] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [errors, setErrors] = useState<ErrorItem[]>([]);
  const [page, setPage] = useState<'config' | 'simulation' | 'report'>('config');

  useWakeLock(shouldConnect);
  const { analyticsOnDownloadRecording } = useGoogleAnalytics({
    shouldConnect,
    unmuteConfig,
  });

  // Check if the backend server is healthy. If we setHealthStatus to null,
  // a "server is down" screen will be shown.
  useEffect(() => {
    if (!backendServerUrl) return;

    setWebSocketUrl(backendServerUrl.toString() + "/v1/realtime");

    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const response = await fetch(`${backendServerUrl}/v1/health`, {
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        if (!response.ok) {
          setHealthStatus({
            connected: "yes_request_fail",
            ok: false,
          });
        }
        const data = await response.json();
        data["connected"] = "yes_request_ok";

        if (data.ok && !data.voice_cloning_up) {
          console.debug("Voice cloning not available, hiding upload button.");
        }
        setHealthStatus(data);
      } catch {
        setHealthStatus({
          connected: "no",
          ok: false,
        });
      }
    };

    checkHealth();
  }, [backendServerUrl]);

  const { sendMessage, lastMessage, readyState } = useWebSocket(
    webSocketUrl || null,
    {
      protocols: ["realtime"],
    },
    shouldConnect
  );

  // Send microphone audio to the server (via useAudioProcessor below)
  const onOpusRecorded = useCallback(
    (opus: Uint8Array) => {
      sendMessage(
        JSON.stringify({
          type: "input_audio_buffer.append",
          audio: base64EncodeOpus(opus),
        })
      );
    },
    [sendMessage]
  );

  const { setupAudio, shutdownAudio, audioProcessor } =
    useAudioProcessor(onOpusRecorded);
  const {
    canvasRef: recordingCanvasRef,
    downloadRecording,
    recordingAvailable,
  } = useRecordingCanvas({
    size: 1080,
    shouldRecord: shouldConnect,
    audioProcessor: audioProcessor.current,
    chatHistory: rawChatHistory,
  });

  const onConnectButtonPress = async () => {
    // If we're not connected yet
    if (!shouldConnect) {
      const mediaStream = await askMicrophoneAccess();
      // If we have access to the microphone:
      if (mediaStream) {
        await setupAudio(mediaStream);
        setShouldConnect(true);
      }
    } else {
      setShouldConnect(false);
      shutdownAudio();
    }
  };

  const onDownloadRecordingButtonPress = () => {
    try {
      downloadRecording(false);
      analyticsOnDownloadRecording();
    } catch (e) {
      if (e instanceof Error) {
        setErrors((prev) => [...prev, makeErrorItem(e.message)]);
      }
    }
  };

  // If the websocket connection is closed, shut down the audio processing
  useEffect(() => {
    if (readyState === ReadyState.CLOSING || readyState === ReadyState.CLOSED) {
      setShouldConnect(false);
      shutdownAudio();
    }
  }, [readyState, shutdownAudio]);

  // Handle incoming messages from the server
  useEffect(() => {
    if (lastMessage === null) return;

    const data = JSON.parse(lastMessage.data);
    if (data.type === "response.audio.delta") {
      const opus = base64DecodeOpus(data.delta);
      const ap = audioProcessor.current;
      if (!ap) return;

      ap.decoder.postMessage(
        {
          command: "decode",
          pages: opus,
        },
        [opus.buffer]
      );
    } else if (data.type === "unmute.additional_outputs") {
      setDebugDict(data.args.debug_dict);
    } else if (data.type === "error") {
      if (data.error.type === "warning") {
        console.warn(`Warning from server: ${data.error.message}`, data);
        // Warnings aren't explicitly shown in the UI
      } else {
        console.error(`Error from server: ${data.error.message}`, data);
        setErrors((prev) => [...prev, makeErrorItem(data.error.message)]);
      }
    } else if (
      data.type === "conversation.item.input_audio_transcription.delta"
    ) {
      // Transcription of the user speech
      setRawChatHistory((prev) => [
        ...prev,
        { role: "user", content: data.delta },
      ]);
    } else if (data.type === "unmute.response.text.delta.ready") {
      // Fast (Real-time) Text Delta from LLM (before TTS sync)
      // We use this for immediate feedback in the transcript
      setRawChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: data.delta },
      ]);
    } else if (data.type === "response.text.delta") {
      // Synced Text-to-speech output.
      // Since we are now using the "Fast" delta above, we should IGNORE this one to avoid duplicates
      // OR we can decide which one to use.
      // For now, let's COMMENT OUT the logic below and rely on the Fast Delta.

      // setRawChatHistory((prev) => [
      //   ...prev,
      //   { role: "assistant", content: " " + data.delta },
      // ]);
    } else {
      const ignoredTypes = [
        "session.updated",
        "response.created",
        "response.text.delta", // Ignored now in favor of fast delta
        "response.text.done",
        "response.audio.done",
        "conversation.item.input_audio_transcription.delta",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.speech_started",
        "unmute.interrupted_by_vad",
        // "unmute.response.text.delta.ready", // Handled above!
        "unmute.response.audio.delta.ready",
      ];
      if (!ignoredTypes.includes(data.type)) {
        console.warn("Received unknown message:", data);
      }
    }
  }, [audioProcessor, lastMessage]);

  // When we connect, we send the initial config (voice and instructions) to the server.
  // Also clear the chat history.
  useEffect(() => {
    if (readyState !== ReadyState.OPEN) return;

    const recordingConsent =
      localStorage.getItem(RECORDING_CONSENT_STORAGE_KEY) === "true";

    setRawChatHistory([]);
    sendMessage(
      JSON.stringify({
        type: "session.update",
        session: {
          instructions: unmuteConfig.instructions,
          voice: unmuteConfig.voice,
          allow_recording: recordingConsent,
        },
      })
    );
  }, [unmuteConfig, readyState, sendMessage]);

  // Disconnect when the voice or instruction changes.
  // TODO: If it's a voice change, immediately reconnect with the new voice.
  useEffect(() => {
    setShouldConnect(false);
    shutdownAudio();
  }, [shutdownAudio, unmuteConfig.voice, unmuteConfig.instructions]);

  if (!healthStatus || !backendServerUrl) {
    return (
      <div className="flex flex-col gap-4 items-center">
        <h1 className="text-xl mb-4">Loading...</h1>
      </div>
    );
  }

  // if (healthStatus && !healthStatus.ok) {
  //    return <CouldNotConnect healthStatus={healthStatus} />;
  // }


  const onStartSimulation = () => {
    setPage('simulation');
  };

  const onStopSimulation = () => {
    setShouldConnect(false);
    shutdownAudio();
    setPage('config');
  };

  const onFinishSimulation = () => {
    setShouldConnect(false);
    shutdownAudio();
    setPage('report');
  };

  if (page === 'config') {
    return (
      <div className="w-full">
        <ErrorMessages errors={errors} setErrors={setErrors} />
        <ConfigPage
          backendServerUrl={backendServerUrl}
          config={unmuteConfig}
          setConfig={setUnmuteConfig}
          onStart={onStartSimulation}
        />
      </div>
    );
  }

  if (page === 'report') {
    return (
      <ReportPage
        chatHistory={rawChatHistory}
        onBack={() => setPage('config')}
      />
    );
  }

  return (
    <div className="w-full">
      <ErrorMessages errors={errors} setErrors={setErrors} />
      {/* The main full-height demo */}
      {/* The main full-height demo */}
      <div className="relative flex w-full h-screen flex-col text-offwhite bg-background overflow-hidden">

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col md:flex-row min-h-0">
          {/* Left Panel: Transcript */}
          <div className="hidden md:flex flex-col w-1/3 lg:w-1/4 border-r border-white/10 bg-black/10">
            <div className="p-4 text-sm font-semibold opacity-50 uppercase tracking-wider border-b border-white/5">
              Live Transcript
            </div>
            <div className="flex-1 min-h-0 p-4">
              <Transcript chatHistory={chatHistory} />
            </div>
          </div>

          {/* Right/Main Panel: Visualizers */}
          <div className="flex-1 flex flex-col items-center justify-center relative p-4">

            {/* Visualizers Container */}
            <div
              className={clsx(
                "w-full flex flex-row items-center justify-center gap-8 md:gap-16",
              )}
            >
              <PositionedAudioVisualizer
                chatHistory={chatHistory}
                role={"assistant"}
                analyserNode={audioProcessor.current?.outputAnalyser || null}
                onCircleClick={onConnectButtonPress}
                isConnected={shouldConnect}
              />
              <div className="flex flex-col items-center gap-2">
                <PositionedAudioVisualizer
                  chatHistory={chatHistory}
                  role={"user"}
                  analyserNode={audioProcessor.current?.inputAnalyser || null}
                  isConnected={shouldConnect}
                  // Visual "Disabled" state if bot is speaking
                  className={
                    // @ts-ignore - debugDict is typed as object
                    debugDict?.conversation_state === "bot_speaking"
                      ? "opacity-30 grayscale pointer-events-none transition-opacity duration-300"
                      : "transition-opacity duration-300"
                  }
                />
                {/* @ts-ignore */}
                {debugDict?.conversation_state === "bot_speaking" && (
                  <div className="text-xs text-white/50 uppercase tracking-widest animate-pulse">
                    Listening...
                  </div>
                )}
              </div>
            </div>

            {/* Mobile Transcript (visible only on small screens) */}
            <div className="md:hidden w-full h-48 mt-8 border-t border-white/10 pt-4">
              <Transcript chatHistory={chatHistory} />
            </div>
          </div>
        </div>

        {showSubtitles && <Subtitles chatHistory={chatHistory} />}

        {/* Controls Footer */}
        <div className="w-full flex flex-col-reverse md:flex-row items-center justify-center px-6 py-6 gap-4 z-10 bg-background border-t border-white/10">
          <SlantedButton
            onClick={onFinishSimulation}
            kind="primary"
            extraClasses="w-full max-w-96 bg-gradium-orange border-gradium-orange hover:bg-orange-600"
          >
            Finish & Generate Report
          </SlantedButton>

          <SlantedButton
            onClick={onConnectButtonPress}
            kind={shouldConnect ? "secondary" : "primary"}
            extraClasses="w-full max-w-96"
          >
            {shouldConnect ? "disconnect" : "connect"}
          </SlantedButton>
          <SlantedButton
            onClick={onStopSimulation}
            kind="secondary"
            extraClasses="w-full max-w-96 bg-red-900 border-red-800"
          >
            stop & exit
          </SlantedButton>
          {microphoneAccess === "refused" && (
            <div className="text-red">
              {"You'll need to allow microphone access to use the demo. " +
                "Please check your browser settings."}
            </div>
          )}
        </div>
      </div>
      {/* Debug stuff, not counted into the screen height */}
      {isDevMode && (
        <div>
          <div className="text-xs w-full overflow-auto">
            <pre
              className="whitespace-pre-wrap break-words"
              dangerouslySetInnerHTML={{
                __html: prettyPrintJson.toHtml(debugDict),
              }}
            ></pre>
          </div>
          <div>Subtitles: press S. Dev mode: press D.</div>
        </div>
      )}
      <canvas ref={recordingCanvasRef} className="hidden" />
    </div>
  );
};

export default Unmute;
