import { useCallback, useEffect, useState } from "react";
import VoiceAttribution from "./VoiceAttribution";
import SquareButton from "./SquareButton";
import Modal from "./Modal";
import { ArrowUpRight } from "lucide-react";
import VoiceUpload from "./VoiceUpload";
// import VoiceUpload from "./VoiceUpload";

export type LanguageCode = "en" | "fr" | "en/fr" | "fr/en";

export type ConstantInstructions = {
  type: "constant";
  text: string;
  language?: LanguageCode;
};

export type Instructions =
  | ConstantInstructions
  | { type: "smalltalk"; language?: LanguageCode }
  | { type: "guess_animal"; language?: LanguageCode }
  | { type: "quiz_show"; language?: LanguageCode };

export type UnmuteConfig = {
  instructions: Instructions;
  voice: string;
  // The backend doesn't care about this, we use it for analytics
  voiceName: string;
  // The backend doesn't care about this, we use it for analytics
  isCustomInstructions: boolean;
};

// Will be overridden immediately by the voices fetched from the backend
export const DEFAULT_UNMUTE_CONFIG: UnmuteConfig = {
  instructions: {
    type: "smalltalk",
    language: "en/fr",
  },
  voice: "barack_demo.wav",
  voiceName: "Missing voice",
  isCustomInstructions: false,
};

export type FreesoundVoiceSource = {
  source_type: "freesound";
  url: string;
  start_time: number;
  sound_instance: {
    id: number;
    name: string;
    username: string;
    license: string;
  };
  path_on_server: string;
};

export type FileVoiceSource = {
  source_type: "file";
  path_on_server: string;
  description?: string;
  description_link?: string;
};

export type VoiceSample = {
  name: string | null;
  comment: string;
  good: boolean;
  instructions: Instructions | null;
  source: FreesoundVoiceSource | FileVoiceSource;
};

const instructionsToPlaceholder = (instructions: Instructions) => {
  if (instructions.type === "constant") {
    return instructions.text;
  } else {
    return (
      {
        smalltalk:
          "Make pleasant conversation. (For this character, the instructions contain dynamically generated parts.)",
        guess_animal:
          "Make the user guess the animal. (For this character, the instructions contain dynamically generated parts.)",
        quiz_show:
          "You're a quiz show host that hates his job. (For this character, the instructions contain dynamically generated parts.)",
        news: "Talk about the latest tech news. (For this character, we fetch the news from the internet dynamically.)",
        dwarf: "Speak like a fantasy dwarf. (For this character, the instructions are long so we don't show them here in full.)",
        customer_support: "Customer support demo. (For this character, the instructions are long so we don't show them here in full.)",
        unmute_explanation:
          "Explain how Unmute works. (For this character, the instructions are long so we don't show them here in full.)",
      }[instructions.type] || ""
    );
  }
};

const fetchVoices = async (
  backendServerUrl: string
): Promise<VoiceSample[]> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(`${backendServerUrl}/v1/voices`, {
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      console.error("Failed to fetch voices:", response.statusText);
      return [];
    }

    const voices = await response.json();
    return voices;
  } catch (error) {
    console.error("Error fetching voices:", error);
    return [];
  }
};

const getVoiceName = (voice: VoiceSample) => {
  return (
    voice.name ||
    (voice.source.source_type === "freesound"
      ? voice.source.sound_instance.username
      : voice.source.path_on_server.slice(0, 10))
  );
};

export interface Scenario {
  id: string;
  label: string;
  description?: string;
}

interface ConfigPageProps {
  backendServerUrl: string | null;
  config: UnmuteConfig;
  setConfig: (config: UnmuteConfig) => void;
  onStart: () => void;
}

import SlantedButton from "./SlantedButton";

export default function ConfigPage({
  backendServerUrl,
  config,
  setConfig,
  onStart,
}: ConfigPageProps) {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>("");

  useEffect(() => {
    if (!backendServerUrl) return;
    try {
      // Ensure no trailing slashes / double slashes
      const cleanBase = backendServerUrl.replace(/\/+$/, "");
      const url = `${cleanBase}/v1/scenarios`;
      console.log("ConfigPage: Base URL raw:", backendServerUrl);
      console.log("ConfigPage: Clean Base:", cleanBase);
      console.log("ConfigPage: Fetching scenarios from:", url);

      fetch(url)
        .then((res) => res.json())
        .then((data) => {
          if (data.scenarios) {
            setScenarios(data.scenarios);
            // Only set default if we haven't selected one yet
            if (data.scenarios.length > 0) {
              setSelectedScenarioId(prev => prev || data.scenarios[0].id);
            }
          }
        })
        .catch(err => console.error("Failed to fetch scenarios", err));
    } catch (e) {
      console.error("URL construction error", e);
    }
  }, [backendServerUrl]);

  const handleScenarioChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedScenarioId(e.target.value);
  };

  const handleStart = () => {
    setConfig({
      ...config,
      instructions: {
        type: "constant",
        text: `[SCENARIO:${selectedScenarioId}] ` + (config.instructions.type === 'constant' ? config.instructions.text : "")
      }
    });
    onStart();
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen w-full bg-background text-offwhite gap-8 p-4">
      <h1 className="text-4xl font-bold mb-8">Simulation Setup</h1>

      <div className="flex flex-col gap-4 w-full max-w-md relative z-10">
        <label className="text-lg font-semibold">Select Patient Scenario</label>
        {scenarios.length === 0 ? (
          <div className="p-3 rounded bg-white/10 border border-white/20 text-gray-400 italic">
            Loading scenarios... (Check backend connection)
          </div>
        ) : (
          <select
            className="p-3 rounded bg-[#333] border border-white/20 text-white w-full cursor-pointer appearance-none hover:border-white/50 transition-colors"
            value={selectedScenarioId}
            onChange={handleScenarioChange}
          >
            {scenarios.map(s => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
        )}

        <label className="text-lg mt-4 font-semibold">Difficulty (1-5)</label>
        <input type="range" min="1" max="5" defaultValue="3" className="w-full" />
        <SlantedButton onClick={handleStart} kind="primary" extraClasses="mt-8 w-full">
          Start Call with Patient
        </SlantedButton>
      </div>
    </div>
  );
}
