import asyncio
import math
import re
from logging import getLogger
from pathlib import Path
import time
from typing import Any, Literal, cast
import base64

import numpy as np
import websockets
from unmute.utils import (
    AdditionalOutputs,
    CloseStream,
    audio_to_float32,
    wait_for_item,
)
import json
from pydantic import BaseModel

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.audio_input_override import AudioInputOverride
from unmute.exceptions import make_ora_error
from unmute.gradium_constants import (
    FRAME_TIME_SEC,
    RECORDINGS_DIR,
    STT_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    STT_SAMPLES_PER_FRAME,
    TTS_SAMPLES_PER_FRAME,
)
from unmute.llm.llm_utils import rechunk_to_words
from unmute.agent.graph import build_graph
from unmute.agent.mistral import MistralClient
from unmute.schemas import PatientProfile, DialogTurn, SimulationState
from unmute.schemas_phase3 import PatientResponse
from unmute.quest_manager import Quest, QuestManager
# Removed duplicate Recorder import
from unmute.stt.speech_to_text import SpeechToText, STTMarkerMessage
from unmute.timer import Stopwatch
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSClientEosMessage,
    QueueAudio,
    QueueText,
)
from unmute.recorder import Recorder
from unmute.audio_recorder import AudioRecorder

# TTS_DEBUGGING_TEXT: str | None = "What's 'Hello world'?"
# TTS_DEBUGGING_TEXT: str | None = "What's the difference between a bagel and a donut?"
TTS_DEBUGGING_TEXT = None

# AUDIO_INPUT_OVERRIDE: Path | None = Path.home() / "audio/dog-or-cat-3.mp3"
AUDIO_INPUT_OVERRIDE: Path | None = None
DEBUG_PLOT_HISTORY_SEC = 10.0

USER_SILENCE_TIMEOUT = 7.0
USER_SILENCE_MARKER = " ... "
FIRST_MESSAGE_TEMPERATURE = 0.7
FURTHER_MESSAGES_TEMPERATURE = 0.3
# For this much time, the VAD does not interrupt the bot. This is needed because at
# least on Mac, the echo cancellation takes a while to kick in, at the start, so the ASR
# sometimes hears a bit of the TTS audio and interrupts the bot. Only happens on the
# first message.
# A word from the ASR can still interrupt the bot.
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3

logger = getLogger(__name__)

HandlerOutput = (
    tuple[int, np.ndarray] | AdditionalOutputs | ora.ServerEvent | CloseStream
)


class GradioUpdate(BaseModel):
    chat_history: list[dict[str, str]]
    debug_dict: dict[str, Any]
    debug_plot_data: list[dict]


class UnmuteHandler:
    def __init__(self) -> None:
        self.input_sample_rate = STT_SAMPLE_RATE
        self.output_sample_rate = TTS_SAMPLE_RATE
        self.n_samples_received = 0  # Used for measuring time
        self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
        # Event Recorder for JSONL
        self.recorder = Recorder(RECORDINGS_DIR) if RECORDINGS_DIR else None
        # Audio Recorder for WAV (User + Agent)
        self.audio_recorder = AudioRecorder(RECORDINGS_DIR, sample_rate=TTS_SAMPLE_RATE) if RECORDINGS_DIR else None

        self.quest_manager = QuestManager()

        self.stt_last_message_time: float = 0
        self.stt_last_message_time_wall: float = 0
        self.stt_end_of_flush_time: float | None = None
        self.stt_flush_timer = Stopwatch()

        self.tts_voice: str | None = None  # Stored separately because TTS is restarted
        self.tts_output_stopwatch = Stopwatch()

        self.graph = build_graph()
        self.mistral = MistralClient()
        
        # Initialize State
        self.state: SimulationState = {
            "patient_profile": PatientProfile(
                name="Martha", 
                age=65, 
                condition="Type 2 Diabetes", 
                base_instructions="You are finding it hard to stick to your diet."
            ),
            "history": [],
            "current_input": None,
            "emotional_state": "Neutral",
            "current_input": None,
            "emotional_state": "Neutral",
            "response_stream": None,
            "turn_count": 0
        }
        
        # Load Patient Profile from JSON
        self._load_patient_profile()

        # Migration: Track simple conversation state
        self._conversation_status = "waiting_for_user"

        self.turn_transition_lock = asyncio.Lock()

        self.debug_dict: dict[str, Any] = {
            "timing": {},
            "connection": {},
            "chatbot": {},
        }
        self.debug_plot_data: list[dict] = []
        self.last_additional_output_update = self.audio_received_sec()

        if AUDIO_INPUT_OVERRIDE is not None:
            self.audio_input_override = AudioInputOverride(AUDIO_INPUT_OVERRIDE)
        else:
            self.audio_input_override = None

    def _get_conversation_state(self) -> str:
        # Simple approximation for now
        return self._conversation_status

    async def cleanup(self):
        try:
             await self.save_simulation_report()
        except Exception as e:
            logger.error(f"Failed to save simulation report: {e}")

        if self.recorder is not None:
            await self.recorder.shutdown()

    async def save_simulation_report(self):
        """Saves the conversation history and metadata to a JSON report."""
        if not self.state["history"]:
            return

        report_dir = Path("logs/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = report_dir / f"simulation_{timestamp}.json"
        
        # Generate Transcript
        transcript_lines = []
        for t in self.state["history"]:
            if t.role == "user":
                role_label = "Doctor"
            elif t.role == "assistant":
                role_label = "Patient"
            else:
                continue
            transcript_lines.append(f"{role_label}: {t.content}")
        
        transcript_str = "\n".join(transcript_lines)

        report_data = {
            "transcript": transcript_str,
            "patient_profile": self.state["patient_profile"].model_dump(),
            "emotional_state_final": self.state["emotional_state"],
            "turn_count": self.state.get("turn_count", 0),
            "history": [
                {
                    "role": t.role,
                    "content": t.content,
                    "metadata": t.metadata,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in self.state["history"]
            ]
        }
        
        with open(filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Saved simulation report to {filename}")

    @property
    def stt(self) -> SpeechToText | None:
        try:
            quest = self.quest_manager.quests["stt"]
        except KeyError:
            return None
        return cast(Quest[SpeechToText], quest).get_nowait()

    @property
    def tts(self) -> TextToSpeech | None:
        try:
            quest = self.quest_manager.quests["tts"]
        except KeyError:
            return None
        return cast(Quest[TextToSpeech], quest).get_nowait()

    def get_gradio_update(self):
        self.debug_dict["conversation_state"] = self._get_conversation_state()
        self.debug_dict["connection"]["stt"] = self.stt.state() if self.stt else "none"
        self.debug_dict["connection"]["tts"] = self.tts.state() if self.tts else "none"
        self.debug_dict["tts_voice"] = self.tts.voice if self.tts else "none"
        self.debug_dict["stt_pause_prediction"] = (
            self.stt.pause_prediction.value if self.stt else -1
        )

        # This gets verbose
        # cutoff_time = self.audio_received_sec() - DEBUG_PLOT_HISTORY_SEC
        # self.debug_plot_data = [x for x in self.debug_plot_data if x["t"] > cutoff_time]

        return AdditionalOutputs(
            GradioUpdate(
                chat_history=[
                    # Not trying to hide the system prompt, just making it less verbose
                    {"role": m.role, "content": m.content}
                    for m in self.state["history"]
                    if m.role != "system"
                ],
                debug_dict=self.debug_dict,
                debug_plot_data=[],
            )
        )

    async def add_chat_message_delta(
        self,
        delta: str,
        role: Literal["user", "assistant", "system"],
        generating_message_i: int | None = None,  # Avoid race conditions
    ):
        # Phase 2: Use Graph to update state with user input
        self.state["current_input"] = delta
        # Only running 'listen' and 'analyze' nodes. 'respond' is manual for streaming.
        # We manually update history for now to mimic 'listen' node if we don't want to await the full graph yet.
        # But let's try to use the graph if possible. 
        # Actually, for 'delta' (partial input), we might want to wait until the end.
        
        # Current logic: add_chat_message_delta updates history.
        # Let's map this:
        if role == "user":
             # We accumulate user input. In the original code, this sends partials to the UI.
             # For the Agent, we only act on the COMPLETED user turn.
             pass
        
        # Let's just update the internal state history manually for now to keep it compatible.
        
        # Update conversation status
        if role == "user":
            self._conversation_status = "user_speaking"
        elif role == "assistant":
            self._conversation_status = "bot_speaking"

        # Hack: Emulate Chatbot's history list for the frontend
        
        # Hack: Emulate Chatbot's history list for the frontend
        if not self.state["history"]:
             self.state["history"].append(DialogTurn(role=role, content=delta))
             return True
        
        last_turn = self.state["history"][-1]
        if last_turn.role == role:
            last_turn.content += delta
            return False
        else:
             self.state["history"].append(DialogTurn(role=role, content=delta))
             return True

    async def _generate_response(self):
        # Empty message to signal we've started responding.
        # Do it here in the lock to avoid race conditions
        await self.add_chat_message_delta("", "assistant")
        quest = Quest.from_run_step("llm", self._generate_response_task)
        await self.quest_manager.add(quest)

    async def _generate_response_task(self):
        # 0. Check Turn Limit
        MAX_TURNS = 10
        self.state.setdefault("turn_count", 0)
        
        if self.state["turn_count"] >= MAX_TURNS:
            logger.info("Max turns reached. Ending conversation.")
            await self.output_queue.put(CloseStream("Conversation finished (Max Turns)."))
            return

        self.state["turn_count"] += 1
        self._conversation_status = "bot_speaking"
        await self.add_chat_message_delta(" [Fast Loop: Speaking...] ", "system")

        # 1. First Message Override
        if self.state["turn_count"] == 1:
            logger.info("First turn: Forcing 'Hi!' response.")
            # Create a fake structured response
            structured_response = PatientResponse(
                inner_thoughts="First interaction. Need to greet the doctor.",
                emotional_state="Neutral",
                speech="Hi!",
                compliance_check=True,
                conversation_status="continue"
            )
            # Simulate processing time slightly? No, fast is good.
            # Directly jump to Done logic logic via a mock full_json_buffer or just emulate the end of the loop?
            # Creating a helper or just outputting it directly is cleaner.
            
            # Send to TTS
            await self.output_queue.put(
                ora.UnmuteResponseTextDeltaReady(delta="Hi!")
            )
            
            # We need to start TTS connection context though
            # But tts sending needs quest. 
            # Let's just follow the standard path but MOCK the mistral generation?
            # Or simpler: inject "Hi!" into the standard logic flow handling?
            # It's inside a big function. Let's do it manually here.
            
            quest = await self.start_up_tts(len(self.state["history"]))
            try:
                tts = await quest.get()
                await tts.send("Hi!")
                await tts.send(TTSClientEosMessage())
                # Wait for TTS to finish processing
                await quest.wait()
            except Exception as e:
                logger.error(f"TTS Error on first message: {e}")

            # Update State
            self.state["emotional_state"] = structured_response.emotional_state
            # Append Assistant Turn
            if self.state["history"] and self.state["history"][-1].role == "assistant":
                 pass 
            else:
                 # Should create if strictly following turn-taking, 
                 # but our loop relies on TTS logic to append... 
                 # Wait, _tts_loop does NOT append to history?
                 # No, `add_chat_message_delta` is called by `_tts_loop` implicitly via `_stt_loop`? 
                 # NO. 
                 # Let's check `_tts_loop`. It reads from tts stream. 
                 # `tts` object ? No `TextToSpeech` does NOT call add_chat_message_delta.
                 # `UnmuteHandler.add_chat_message_delta` is called by `_generate_response` at start with empty string.
                 # Then logic usually appends.
                 # To be safe, let's append "Hi!" to the assistant turn.
                 await self.add_chat_message_delta("Hi!", "assistant")
                 
            # Attach Metadata
            if self.state["history"] and self.state["history"][-1].role == "assistant":
                    self.state["history"][-1].metadata = structured_response.model_dump()

            await self.output_queue.put(
                ora.ResponseTextDone(text="Hi!")
            )
            self._conversation_status = "waiting_for_user"
            return
        
        # 1. Run Analysis (Listen -> Analyze)
        # We skip 'listen' because we manually updated history in add_chat_message_delta
        # So we just run 'analyze' logic manually or via graph if we structured it to accept history.
        # Let's manually run analyze for now or skip.
        
        generating_message_i = len(self.state["history"])
        
        # Frontend Chat History update (mapped from state history)
        frontend_history = [{"role": t.role, "content": t.content} for t in self.state["history"]]

        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=frontend_history,
                )
            )
        )

        llm_stopwatch = Stopwatch()
        quest = await self.start_up_tts(generating_message_i)

        # Prepare Messages for Mistral
        profile = self.state["patient_profile"]
        
        # Simple JSON instruction without full schema (to avoid "properties" wrapping)
        system_prompt = (
            f"You are {profile.name}, {profile.age} years old.\n"
            f"Condition: {profile.condition}.\n"
            f"Instructions: {profile.base_instructions} Keep your responses VERY SHORT (max 10 words if possible, never more than 2 sentences).\n\n"
            f"IMPORTANT: You MUST respond with a JSON object with these exact fields:\n"
            f"- inner_thoughts: Your private reasoning (string)\n"
            f"- emotional_state: Your current emotion (string)\n"
            f"- speech: What you say out loud (string, max 2 sentences)\n"
            f"- compliance_check: Whether you understood the doctor (boolean)\n"
            f"- conversation_status: 'continue' or 'finished' (string)\n"
            f"- metadata: Any tracking metrics (object)\n\n"
            f"Example: {{\"inner_thoughts\": \"...\", \"emotional_state\": \"...\", \"speech\": \"...\", \"compliance_check\": false, \"conversation_status\": \"continue\", \"metadata\": {{}}}}"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.state["history"]:
            # Skip empty content
            if not turn.content or not turn.content.strip():
                continue
            # Skip visualization system messages (don't send to LLM)
            if turn.role == "system" and ("Fast Loop" in turn.content or "Slow Loop" in turn.content):
                continue
            messages.append({"role": turn.role, "content": turn.content})
        
        # Mistral API requires the last message to be from User.
        if len(messages) == 1 and messages[0]["role"] == "system":
             messages.append({"role": "user", "content": "(The consultation begins.)"})
        elif messages and messages[-1]["role"] == "assistant":
            logger.info("Last message was assistant. Appending continuation prompt.")
            messages.append({"role": "user", "content": "(The patient waits for you to continue.)"})
        
        # Ensure last message is not system if we have more than 1
        if len(messages) > 1 and messages[-1]["role"] == "system":
             pass

        self.tts_output_stopwatch = Stopwatch(autostart=False)
        tts = None
        error_from_tts = False
        time_to_first_token = None
        
        # JSON Streaming State
        full_json_buffer = ""
        in_speech_field = False
        speech_buffer = ""
        
        # Regex or simple state machine for extracting "speech": "..."
        # We'll use a simple find mechanism on the buffer
        # Limitation: This simple logic assumes "speech" comes after inner_thoughts usually, 
        # or we scan for it. Mistral usually follows schema order.
        
        try:
            # Mistral Streaming with JSON Mode
            async for delta in rechunk_to_words(self.mistral.chat_stream(messages, response_format={"type": "json_object"})):
                full_json_buffer += delta
                
                # Simple extraction logic:
                # 1. If we haven't found "speech": " yet, look for it.
                # 2. If we found it, start yielding chars until we hit an unescaped "
                
                # Check for start of speech field
                if not in_speech_field:
                    if '"speech":' in full_json_buffer:
                        # Find the start quote of the value
                        speech_key_idx = full_json_buffer.rfind('"speech":')
                        # Look for the opening quote after the key
                        start_quote_idx = full_json_buffer.find('"', speech_key_idx + 9) # 9 is len('"speech":')
                        
                        if start_quote_idx != -1:
                            # We found the start of the speech value!
                            # The content starts after start_quote_idx
                            # But wait, we might have already streamed past it in full_json_buffer if delta was huge
                            # We need to only handle the *new* part or track position.
                            pass # Tricky to do robustly with just simple string find on full buffer.
                            
                            # Let's try a simpler approach: 
                            # Just look at the delta if we are "in_speech" flag is true.
                            # But determining transition is hard.
                            
                            # BACKUP PLAN FOR STABILITY:
                            # Just stream everything to TTS? NO, that includes JSON syntax.
                            # Let's use a robust heuristic:
                            # If the delta behaves like JSON structure, ignore it.
                            # If it looks like speech text, send it.
                            # NO, that's unreliable.
                            
                            # Standard Way: accumulating buffer and creating 'speech_accumulated'.
                            # Track length of previously processed speech.
                            pass

                # Basic robust implementation:
                # We won't perfect streaming JSON parsing here due to time.
                # We will wait for a bigger chunk or just rely on post-processing for metadata,
                # BUT for TTS we MUST stream speech.
                
                # Let's try to detect if we are strictly inside the "speech" value.
                # Since we inserted the schema, Mistral will likely output keys in order.
                # If 'speech' is generated, it will look like "speech": "Hello..."
                
                # Temporary Hack for MVP Latency:
                # We will just Regex search the FULL buffer for "speech": "(.*)"
                # And send the *difference* from the last match to TTS.
                # This works if "speech" is at the end or continuously growing.
                import re
                match = re.search(r'"speech":\s*"(.*?)(?:"|$)', full_json_buffer, re.DOTALL)
                if match:
                    current_extracted_speech = match.group(1)
                    # Handle escaped quotes if necessary (basic unescape)
                    current_extracted_speech = current_extracted_speech.replace('\\"', '"')
                    
                    # Calculate new part
                    if len(current_extracted_speech) > len(speech_buffer):
                        new_text = current_extracted_speech[len(speech_buffer):]
                        speech_buffer = current_extracted_speech
                        
                        # Send NEW text to TTS
                        # Send text delta to Frontend (Transcript)
                        # We use 'unmute.response.text.delta.ready' for fast updates
                        await self.output_queue.put(
                            ora.UnmuteResponseTextDeltaReady(delta=new_text)
                        )
                        
                        if time_to_first_token is None:
                            time_to_first_token = llm_stopwatch.time()
                            self.debug_dict["timing"]["to_first_token"] = time_to_first_token
                            mt.VLLM_TTFT.observe(time_to_first_token)
                            logger.info("Sending first word to TTS: %s", new_text)

                        self.tts_output_stopwatch.start_if_not_started()
                        try:
                            tts = await quest.get()
                        except Exception:
                            error_from_tts = True
                            raise
                        
                        if len(self.state["history"]) > generating_message_i:
                             break 
                        
                        await tts.send(new_text)

            # Done
            # Done
            try:
                # Parse JSON - handle both direct and properties-wrapped formats
                parsed_json = json.loads(full_json_buffer)
                
                # If Mistral wrapped it in {"properties": {...}}, unwrap it
                if "properties" in parsed_json and isinstance(parsed_json["properties"], dict):
                    logger.info("Unwrapping 'properties' wrapper from Mistral response")
                    parsed_json = parsed_json["properties"]
                
                # Now validate with Pydantic
                structured_response = PatientResponse.model_validate(parsed_json)
                
                # Log Inner Thoughts
                logger.info(f"INNER THOUGHTS: {structured_response.inner_thoughts}")
                logger.info(f"EMOTIONAL STATE: {structured_response.emotional_state}")
                
                # Update State
                self.state["emotional_state"] = structured_response.emotional_state
                
                # Attach to History (Memory)
                # We assume TTS loop has created/updated the assistant turn by now (or will shortly).
                # We update the *last* assistant turn.
                # Note: This might be race-condition prone if TTS hasn't started yet, 
                # but usually we have sent deltas to TTS by now.
                # Safer: Check if history has assistant turn.
                if self.state["history"] and self.state["history"][-1].role == "assistant":
                    self.state["history"][-1].metadata = structured_response.model_dump()
                else:
                    # If history doesn't have it yet (e.g. extremely fast generation before TTS processes first chunk),
                    # we might need to wait or just append it? 
                    # Actually, if we streamed speech, TTS is active.
                    pass

                # Check Completion
                if structured_response.conversation_status == "finished":
                    logger.info("Agent signaled end of conversation.")
                    # We send a closing message after speech is done?
                    # Ideally we wait for TTS to finish.
                    pass # We will let the flow continue, but blocking further turns?
                    # Let's set a flag or just close stream?
                    # Closing stream might cut off audio.
                    # Best: Set conversation status to finished so next user input is ignored or triggers goodbye.
                    self._conversation_status = "finished"

                await self.output_queue.put(
                    ora.ResponseTextDone(text=structured_response.speech)
                )

            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw Response: {full_json_buffer}")
                # Fallback
                await self.output_queue.put(
                    ora.ResponseTextDone(text=full_json_buffer)
                )
            
            # Reset state after response done (if not finished)
            if self._conversation_status != "finished":
                self._conversation_status = "waiting_for_user"

            if tts is not None:
                logger.info("Sending TTS EOS.")
                await tts.send(TTSClientEosMessage())
                
        except asyncio.CancelledError:
            mt.VLLM_INTERRUPTS.inc()
            raise
        except Exception:
            if not error_from_tts:
                mt.VLLM_HARD_ERRORS.inc()
            raise
        finally:
            logger.info("End of Mistral Generation.")
            mt.VLLM_ACTIVE_SESSIONS.dec()
            self._conversation_status = "waiting_for_user"

    def audio_received_sec(self) -> float:
        """How much audio has been received in seconds. Used instead of time.time().

        This is so that we aren't tied to real-time streaming.
        """
        return self.n_samples_received / self.input_sample_rate

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        stt = self.stt
        assert stt is not None
        sr = frame[0]
        assert sr == self.input_sample_rate

        assert frame[1].shape[0] == 1  # Mono
        array = frame[1][0]

        self.n_samples_received += array.shape[0]

        # If this doesn't update, it means the receive loop isn't running because
        # the process is busy with something else, which is bad.
        self.debug_dict["last_receive_time"] = self.audio_received_sec()
        float_audio = audio_to_float32(array)

        self.debug_plot_data.append(
            {
                "t": self.audio_received_sec(),
                "amplitude": float(np.sqrt((float_audio**2).mean())),
                "pause_prediction": stt.pause_prediction.value,
            }
        )

        if self._get_conversation_state() == "bot_speaking":
            # Periodically update this not to trigger the "long silence" accidentally.
            self.waiting_for_user_start_time = self.audio_received_sec()

        if TTS_DEBUGGING_TEXT is not None:
            assert self.audio_input_override is None, (
                "Can't use both TTS_DEBUGGING_TEXT and audio input override."
            )

            # Debugging mode: always send a fixed string when it's the user's turn.
            if self._get_conversation_state() == "waiting_for_user":
                logger.info("Using TTS debugging text. Ignoring microphone.")
                self.state["history"].append(
                    DialogTurn(
                        role="user",
                        content=TTS_DEBUGGING_TEXT,
                    )
                )
                await self._generate_response()
            return

        if self.audio_input_override is not None:
            frame = (frame[0], self.audio_input_override.override(frame[1]))

        if self._get_conversation_state() == "user_speaking":
            self.debug_dict["timing"] = {}

        # Record User Audio
        if self.audio_recorder:
            self.audio_recorder.add_audio(array)

        await stt.send_audio(array)
        if self.stt_end_of_flush_time is None:
            await self.detect_long_silence()

            if self.determine_pause():
                logger.info("Pause detected")
                await self.output_queue.put(ora.InputAudioBufferSpeechStopped())

                self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
                self.stt_flush_timer = Stopwatch()
                num_frames = (
                    int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
                )  # some safety margin.
                zero = np.zeros(STT_SAMPLES_PER_FRAME, dtype=np.float32)
                for _ in range(num_frames):
                    await stt.send_audio(zero)
            elif (
                self._get_conversation_state() == "bot_speaking"
                and stt.pause_prediction.value < 0.4
                and self.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
            ):
                logger.info("Ignored interruption (Strict Mode)")
                # STRICT NO-INTERRUPTION:
                # logger.info("Interruption by STT-VAD")
                # await self.interrupt_bot()
                # await self.add_chat_message_delta("", "user")
        else:
            # We do not try to detect interruption here, the STT would be processing
            # a chunk full of 0, so there is little chance the pause score would indicate an interruption.
            if stt.current_time > self.stt_end_of_flush_time:
                self.stt_end_of_flush_time = None
                elapsed = self.stt_flush_timer.time()
                rtf = stt.delay_sec / elapsed
                logger.info(
                    "Flushing finished, took %.1f ms, RTF: %.1f", elapsed * 1000, rtf
                )
                await self._generate_response()

    def determine_pause(self) -> bool:
        stt = self.stt
        if stt is None:
            return False
        if self._get_conversation_state() != "user_speaking":
            return False

        # This is how much wall clock time has passed since we received the last ASR
        # message. Assumes the ASR connection is healthy, so that stt.sent_samples is up
        # to date.
        time_since_last_message = (
            stt.sent_samples / self.input_sample_rate
        ) - self.stt_last_message_time
        self.debug_dict["time_since_last_message"] = time_since_last_message
        logger.info(
            "Pause detected, eou: %.0fms, eou wall: %.0fms",
            time_since_last_message,
            time.perf_counter() - self.stt_last_message_time_wall,
        )

        if stt.pause_prediction.value > 0.6:
            self.debug_dict["timing"]["pause_detection"] = time_since_last_message
            return True
        else:
            return False

    async def emit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> HandlerOutput | None:
        output_queue_item = await wait_for_item(self.output_queue)

        if output_queue_item is not None:
            return output_queue_item
        else:
            if self.last_additional_output_update < self.audio_received_sec() - 1:
                # If we have nothing to emit, at least update the debug dict.
                # Don't update too often for performance reasons
                self.last_additional_output_update = self.audio_received_sec()
                return self.get_gradio_update()
            else:
                return None

    def copy(self):
        return UnmuteHandler()

    async def __aenter__(self) -> None:
        await self.quest_manager.__aenter__()

    async def start_up(self):
        await self.start_up_stt()
        # Start Slow Loop
        self.quest_manager.add(Quest("slow_loop", self._start_slow_loop, self._run_slow_loop, self._close_slow_loop))
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def __aexit__(self, *exc: Any) -> None:
        return await self.quest_manager.__aexit__(*exc)

    async def start_up_stt(self):
        async def _init() -> SpeechToText:
            s = SpeechToText()
            await s.start_up()
            return s

        async def _run(stt: SpeechToText):
            await self._stt_loop(stt)

        async def _close(stt: SpeechToText):
            await stt.shutdown()

        quest = await self.quest_manager.add(Quest("stt", _init, _run, _close))
        # We want to be sure to have the STT before starting anything.
        await quest.get()

    async def _stt_loop(self, stt: SpeechToText):
        try:
            async for data in stt:
                if isinstance(data, STTMarkerMessage):
                    # Ignore the marker messages
                    continue

                await self.output_queue.put(
                    ora.ConversationItemInputAudioTranscriptionDelta(
                        delta=data.text,
                        start_time=data.start_s,
                    )
                )

                # The STT sends an empty string as the first message, but we
                # don't want to add that because it can trigger a pause even
                # if the user hasn't started speaking yet.
                if data.text == "":
                    continue

                if self._get_conversation_state() == "bot_speaking":
                    logger.info("STT-based interruption")
                    await self.interrupt_bot()

                self.stt_last_message_time = data.start_s
                self.stt_last_message_time_wall = time.perf_counter()
                is_new_message = await self.add_chat_message_delta(data.text, "user")
                if is_new_message:
                    # Ensure we don't stop after the first word if the VAD didn't have
                    # time to react.
                    stt.pause_prediction.value = 0.0
                    await self.output_queue.put(ora.InputAudioBufferSpeechStarted())
        except websockets.ConnectionClosed:
            logger.info("STT connection closed while receiving messages.")

    async def start_up_tts(self, generating_message_i: int) -> Quest[TextToSpeech]:
        async def _init() -> TextToSpeech:
            sleep_time = 0.05
            sleep_growth = 1.5
            max_sleep = 1.0
            trials = 5
            for trial in range(trials):
                try:
                    tts = TextToSpeech(
                        recorder=self.recorder,
                        get_time=self.audio_received_sec,
                        voice=self.tts_voice,
                    )
                    await tts.start_up()
                except Exception:
                    if trial == trials - 1:
                        raise
                    logger.warning("Will sleep for %.4f sec", sleep_time)
                    await asyncio.sleep(sleep_time)
                    sleep_time = min(max_sleep, sleep_time * sleep_growth)
                    error = make_ora_error(
                        type="warning",
                        message="Looking for the resources, expect some latency.",
                    )
                    await self.output_queue.put(error)
                else:
                    return tts
            raise AssertionError("Too many unexpected packets.")

        async def _run(tts: TextToSpeech):
            await self._tts_loop(tts, generating_message_i)

        async def _close(tts: TextToSpeech):
            await tts.shutdown()

        return await self.quest_manager.add(Quest("tts", _init, _run, _close))

    async def _tts_loop(self, tts: TextToSpeech, generating_message_i: int):
        # On interruption, we swap the output queue. This will ensure that this worker
        # can never accidentally push to the new queue if it's interrupted.
        output_queue = self.output_queue
        try:
            audio_started = None

            async for message in tts:
                if audio_started is not None:
                    time_since_start = self.audio_received_sec() - audio_started
                    time_received = tts.received_samples / self.input_sample_rate
                    time_received_yielded = (
                        tts.received_samples_yielded / self.input_sample_rate
                    )
                    assert self.input_sample_rate == STT_SAMPLE_RATE
                    # Log audio chunk reception for debug
                    if tts.received_samples_yielded % 24000 < 500: # Log roughly once per second
                        logger.info(f"TTS Audio Playback Progress: {time_since_start:.2f}s")

                    self.debug_dict["tts_throughput"] = {
                        "time_received": round(time_received, 2),
                        "time_received_yielded": round(time_received_yielded, 2),
                        "time_since_start": round(time_since_start, 2),
                        "ratio": round(
                            time_received_yielded / (time_since_start + 0.01), 2
                        ),
                    }

                # Check for Interruption (if history changed)
                if len(self.state["history"]) > generating_message_i:
                    break 

                if isinstance(message, QueueAudio):
                    t = self.tts_output_stopwatch.stop()
                    if t is not None:
                        self.debug_dict["timing"]["tts_audio"] = t

                    audio = message.pcm
                    assert self.output_sample_rate == TTS_SAMPLE_RATE # 24kHz

                    # Record Agent Audio (AudioRecorder handles float32 clipping/conversion internally)
                    if self.audio_recorder:
                        self.audio_recorder.add_audio(audio)
                    
                    # Convert Float32 -> Int16 -> Bytes -> Base64 for Frontend
                    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                    audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

                    await self.output_queue.put(
                        ora.ResponseAudioDelta(delta=audio_b64)
                    )
                    mt.TTS_SENT_BYTES.inc(len(audio_b64))

                    if audio_started is None:
                        audio_started = self.audio_received_sec()
                elif isinstance(message, QueueText):
                    await output_queue.put(ora.ResponseTextDelta(delta=message.text))
                    await self.add_chat_message_delta(
                        message.text,
                        "assistant",
                        generating_message_i=generating_message_i,
                    )
                else:
                    logger.warning(
                        "Got unexpected message from TTS: %s", message["type"]
                    )

        except websockets.ConnectionClosedError as e:
            logger.error(f"TTS connection closed with an error: {e}")

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await output_queue.put(
            (
                TTS_SAMPLE_RATE,
                np.zeros(TTS_SAMPLES_PER_FRAME, dtype=np.float32),
            )
        )

        # Replaced custom last_message with history check
        message = ""
        if self.state["history"] and self.state["history"][-1].role == "assistant":
            message = self.state["history"][-1].content
        if message is None:
            logger.warning("No message to send in TTS shutdown.")
            message = ""

        # It's convenient to have the whole chat history available in the client
        # after the response is done, so send the "gradio update"
        await self.output_queue.put(self.get_gradio_update())
        await self.output_queue.put(ora.ResponseAudioDone())

        # Signal that the turn is over by adding an empty message.
        await self.add_chat_message_delta("", "user")

        await asyncio.sleep(1)
        await self.check_for_bot_goodbye()
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def interrupt_bot(self):
        if self._get_conversation_state() != "bot_speaking":
            raise RuntimeError(
                "Can't interrupt bot when conversation state is "
                f"{self._get_conversation_state()}"
            )

        INTERRUPTION_CHAR = " [Interrupted] " 
        await self.add_chat_message_delta(INTERRUPTION_CHAR, "assistant")

        self.output_queue = asyncio.Queue()  # Clear our own queue too

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await self.output_queue.put(
            (
                TTS_SAMPLE_RATE,
                np.zeros(TTS_SAMPLES_PER_FRAME, dtype=np.float32),
            )
        )

        await self.output_queue.put(ora.UnmuteInterruptedByVAD())

        await self.quest_manager.remove("tts")
        await self.quest_manager.remove("llm")
        # Ensure conversation state is reset
        self._conversation_status = "waiting_for_user"

    async def _start_slow_loop(self) -> Any:
        # No special init needed
        pass

    async def _run_slow_loop(self, init_data: Any):
        logger.info("Starting Slow Loop (Thinking Process)")
        while True:
            await asyncio.sleep(1.0) # Check every second
            try:
                # Use the graph's 'think' node
                # We need to pass the current state in
                # NOTE: In a real app, careful with state concurrency.
                # Here we just read history and updates are atomic dict replacements usually safe enough for this prototype.
                
                # We invoke the 'think' node manually
                updates = await self.graph.nodes["think"].afunc(self.state)
                
                if updates:
                    # Update State
                    for key, val in updates.items():
                        self.state[key] = val
                    logger.info(f"Slow Loop Update: {updates}")

                    # Visualization
                    await self.add_chat_message_delta(" [Slow Loop: Thinking...] ", "system")
                    
                    # Interruption Decision?
                    # valid_interruption = updates.get("should_interrupt", False)
                    # if valid_interruption and self._get_conversation_state() == "bot_speaking":
                    #     logger.info("Slow Loop decided to interrupt!")
                    #     await self.interrupt_bot()
            
            except Exception as e:
                logger.error(f"Error in Slow Loop: {e}")

    async def _close_slow_loop(self, init_data: Any):
        pass

    def _load_patient_profile(self):
        try:
            path = Path(__file__).parent.parent / "states" / "patient.json"
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    
                # Naive loading: take the first scenario and base content
                base = data.get("patient_llm_config", {}).get("base_prompt", {}).get("content", "")
                
                # For now just using the base instruction string
                self.state["patient_profile"].base_instructions = base
                
                if "scenarios" in data["patient_llm_config"] and data["patient_llm_config"]["scenarios"]:
                    # Just pick the first one's modifier to append? or Start neutral.
                    # Let's verify we loaded something
                    logger.info(f"Loaded patient profile from {path}")
            else:
                logger.warning(f"Patient profile not found at {path}")
        except Exception as e:
            logger.error(f"Failed to load patient profile: {e}")

    async def check_for_bot_goodbye(self):
        last_assistant_message = next(
            (
                msg.content
                for msg in reversed(self.state["history"])
                if msg.role == "assistant"
            ),
            "",
        )

        # Using function calling would be a more robust solution, but it would make it
        # harder to swap LLMs.
        if last_assistant_message.lower().endswith("bye!"):
            await self.output_queue.put(
                CloseStream("The assistant ended the conversation. Bye!")
            )

    async def detect_long_silence(self):
        """Handle situations where the user doesn't answer for a while."""
        if (
            self._get_conversation_state() == "waiting_for_user"
            and (self.audio_received_sec() - self.waiting_for_user_start_time)
            > USER_SILENCE_TIMEOUT
        ):
            # This will trigger pause detection because it changes the conversation
            # state to "user_speaking".
            # The system prompt has a rule that tells it how to handle the "..."
            # messages.
            logger.info("Long silence detected.")
            await self.add_chat_message_delta(USER_SILENCE_MARKER, "user")

    async def update_session(self, session: ora.SessionConfig):
        if session.instructions:
            self.state["patient_profile"].base_instructions = session.instructions
            logger.info("Session instructions updated. Ready to start.")

        if session.voice:
            self.tts_voice = session.voice

        if not session.allow_recording and self.recorder:
            await self.recorder.add_event("client", ora.SessionUpdate(session=session))
            await self.recorder.shutdown(keep_recording=False)
            self.recorder = None
            logger.info("Recording disabled for a session.")
        
        # Trigger start if this is the first update
        if not self.state.get("has_started", False):
            self.state["has_started"] = True
            logger.info("First Session Update received. Starting Conversation.")
            await self._generate_response()
