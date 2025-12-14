import { Frank_Ruhl_Libre } from "next/font/google";
import Modal from "./Modal";
import { ArrowUpRight } from "lucide-react";

const frankRuhlLibre = Frank_Ruhl_Libre({
  weight: "400",
  subsets: ["latin"],
});

const ShortExplanation = () => {
  return (
    <p className="text-xs text-right">
      Speak to an AI using our new low-latency Speech-to-Text and Text-to-Speech models.
    </p>
  );
};

const UnmuteHeader = () => {
  return (
    <div className="flex flex-col gap-2 py-2 md:py-8 items-end max-w-80 md:max-w-60 lg:max-w-80">
      {/* kyutaiLogo */}
      <h1 className={`text-3xl ${frankRuhlLibre.className}`}>Unmute</h1>
      <div className="flex items-center gap-2 -mt-1 text-xs">
        by
        <a href="https://gradium.ai" target="_blank" rel="noopener">gradium.ai</a>
      </div>
      <ShortExplanation />
      <Modal
        trigger={
          <span className="flex items-center gap-1 text-lightgray">
            More info <ArrowUpRight size={24} />
          </span>
        }
        forceFullscreen={true}
      >
        <div className="flex flex-col gap-3">
          <p>
            This is a cascaded system made by Gradium: our speech-to-text
            transcribes what you say, an LLM (we use Mistral Small 24B)
            generates the text of the response, and we then use our
            text-to-speech model to say it out loud.
          </p>
        </div>
      </Modal>
    </div>
  );
};

export default UnmuteHeader;
