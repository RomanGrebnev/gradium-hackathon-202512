## Specific "Vibe Coding" Prompts

*Use these prompts to drive the AI (Cursor/Windsurf/Antigravity) during development.*

**Prompt for Phase 1 (Logic):**
> "I need to implement the simulation logic using LangGraph and Mistral. Please create a file `agent_logic.py`. It should import the Pydantic models from `schemas.py`. Create a graph where the state holds the conversation history. The node `generate_patient_response` should call Mistral. Depending on the `emotional_archetype` in the state, inject specific instructions into the system prompt (e.g., if 'Hysteric', use caps lock occasionally and refuse to listen)."

**Prompt for Phase 2 (The Async Loop):**
> "I am connecting the backend to the Gradium UI. I need an async function `handle_audio_input(audio_path)`. This function needs to Transcribe (STT), send to our LangGraph agent, and then generate Audio (TTS). Crucial: Ensure the response includes a timestamp so we can track latency. If the generation takes longer than 2 seconds, yield a 'filler' audio file first."

**Prompt for Phase 3 (Evaluation):**
> "Create an evaluator function that takes the full conversation transcript. Use Mistral to grade the doctor based on the 'SPIKES' protocol for breaking bad news. Return the result strictly as the `AssessmentScore` JSON object defined in our schemas."