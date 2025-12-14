# Phase 1 Specification: The Brain (Backend & Logic)

**Goal**: Build the LangGraph agent that simulates the teenage patient ("Alex"). This phase implements the core logic, state management, and personality simulation, running as a text-only interaction first.

**Source of Truth**: Derived from `mission.md` (Section 3: Phase 1).

## 1. Profile Generation (`unmute/core/profiles.py`)
Implement the logic to generate a full `PatientProfile` from basic inputs.

*   **Function**: `generate_profile(age: int, scenario_id: str) -> PatientProfile`
*   **Logic**:
    *   Load `states/patient.json`.
    *   Find the scenario matching `scenario_id` (e.g., "defensive_angry").
    *   Initialize `PatientProfile` with the given `age` and the `prompt_modifier` from the JSON.
    *   The `system_prompt` for the agent should construct the full persona by combining `base_prompt` + `scenario.prompt_modifier`.

## 2. Agent Logic (`unmute/agent/graph.py`)
Implement the LangGraph state machine.

### A. State Definition
Reuse `SimulationState` from `schema.py` as the graph state.

### B. Nodes
1.  **`node_listener`**:
    *   Updates `SimulationState` with the Doctor's input text (simulated STT for now).
2.  **`node_sentiment_analysis`**:
    *   **Input**: Doctor's text.
    *   **Action**: Fast LLM call or rule-based check.
    *   **Output**: Updates `emotional_state` (e.g., "Neutral" -> "Defensive" if text contains jargon).
3.  **`node_responder`**:
    *   **Input**: `SimulationState` (History + Emotional State).
    *   **Action**: Calls **Mistral API** directly (using `MistralStream` or similar wrapper) with `PATIENT_SYSTEM_PROMPT`.
    *   **Output**: Generates `PatientResponse` text stream.
    *   **Note**: Do NOT use `VLLMStream` or `autoselect_model` to avoid valid URL checks on startup.

### C. Graph Flow
*   `START` -> `node_listener` -> `node_sentiment_analysis` -> `node_responder` -> `END`

## 3. Testing Harness (`scripts/test_agent.py`)
Create a CLI script to chat with the agent in the terminal.

*   **Usage**: `uv run python scripts/test_agent.py --mood Denial`
*   **Flow**:
    1.  User types input.
    2.  Agent processes and prints response.
    3.  Loop continues until User types "/quit".
    4.  Prints the full `SimulationState` dump at the end for inspection.

## 4. Verification Plan
*   **Unit Test**: `tests/test_profile_generator.py` ensure profile creation works.
*   **Functional Test**: Run `scripts/test_agent.py` and verify the agent stays in character (e.g., acts like a teenager, gets confused by jargon).
