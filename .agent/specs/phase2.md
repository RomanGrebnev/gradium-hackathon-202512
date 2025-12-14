# Phase 2 Specification: The Voice Loop (Integration)

**Goal**: Connect the "Brain" (LangGraph) to the Gradium UI (Mouth/Ears), enabling real-time voice interaction with latency optimization.

**Source of Truth**: Derived from `mission.md` (Section 3: Phase 2).

## 1. UI Architecture (`unmute/ui/app.py`)
Build the Gradium (Gradio-based) application with state persistence.

### A. Page 1: Configuration
*   **Components**:
    *   Dropdown: `Scenario` (Populated from `patient_llm_config.scenarios` in `patient.json`, showing `label`).
    *   Slider: `Difficulty` (1-5).
    *   Button: `Start Simulation`
*   **Action**:
    *   On "Start", call `generate_profile` with selected `scenario_id`.
    *   Initialize `SimulationState`.
    *   Transition to Page 2.

### B. Page 2: The Simulation
*   **Components**:
    *   `AudioRecorder`: Captures Doctor's voice.
    *   `AudioPlayer`: Plays Patient's response.
    *   `ChatLog` (Optional): Displays text transcript for debugging.
    *   `StatusIndicator`: Shows "Listening...", "Thinking...", "Speaking".

## 2. Audio Pipeline & Latency (`unmute/core/audio.py`)

### A. Async Audio Handler
*   **Function**: `handle_audio_input(audio_chars: bytes, state: State)`
*   **Flow**:
    1.  **STT**: Transcribe audio (using Gradium's built-in or fast Whisper).
    2.  **Streaming**: Send text input to LangGraph Agent (`unmute/agent/graph.py`).
    3.  **TTS**:
        *   **Sentence-Level Streaming**: As soon as `node_responder` yields chunks, aggregate them.
        *   **Integration**: Replace `self.chatbot` in `UnmuteHandler` with `self.agent`.
        *   **Audio Fillers**: If processing time > 1.5s, yield a pre-recorded filler ("Um...", "Let me think...") *before* the actual response.

## 3. Gradium Integration
*   Use `gr.State` to persist `SimulationState` across interactions.
*   Ensure the audio loop is non-blocking (async).

## 4. Verification Plan
*   **Latency Check**: Measure time from "Input Done" to "Audio Start". Target: < 3 seconds.
*   **State Persistence**: Verify that `PatientProfile` created on Page 1 is correctly accessed on Page 2.
*   **Filler Logic**: Force a slow response (artificial delay) and verify the "Um..." filler plays.
