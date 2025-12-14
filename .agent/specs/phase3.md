# Phase 3: Robust Conversation & Structured Evaluation

## Goal
Transition the agent from free-form text generation to **Structured Output** using Pydantic schemas. This ensures robust role adherence (preventing leaking instructions), enables "inner thought" reasoning separate from speech, and allows for precise metadata collection for post-simulation evaluation.

## 1. Structured Output (The "Mind" of the Agent)

### Problem
Currently, the LLM outputs unstructured text. This leads to:
*   **Instruction Leaks**: "Scenario: ..." or "I am acting as..." appearing in speech.
*   **Monologues**: Difficulty enforcing strict length constraints.
*   **Hidden Context Loss**: No way to track *why* the agent said something (inner thoughts) without them speaking it.

### Solution: `PatientResponse` Schema
We will enforce a strict JSON output structure (via Mistral's structured output or tool calling mode) using Pydantic.

```python
from pydantic import BaseModel, Field

class PatientResponse(BaseModel):
    inner_thoughts: str = Field(
        description="Your internal reasoning process. Analyze the doctor's words, your current pain level, and emotional reaction. This is NOT spoken."
    )
    emotional_state: str = Field(
        description="Current detailed emotional state (e.g., 'Defensive and slightly confused', 'Overwhelmed')."
    )
    speech: str = Field(
        description="The actual words spoken to the doctor. Must be short (max 2-3 sentences), natural, and adhere to the persona."
    )
    compliance_check: bool = Field(
        description="True if the doctor's explanation was understood and accepted, False otherwise."
    )
    metadata: dict = Field(
        default={},
        description="Any additional metric to track (e.g., 'doctor_empathy_perceived': 1-10)."
    )
```

### Implementation Details
*   **`MistralClient`**: Update to support `response_format={"type": "json_object"}` or tool use, enforcing the schema.
*   **Handler Logic**:
    *   **Parse** the JSON response.
    *   **Log** `inner_thoughts` and `emotional_state` to the console/transcript (hidden from audio).
    *   **Send** ONLY `speech` to the TTS engine.

## 2. Conversation Memory & Evaluation

### Structured History
Instead of a simple list of strings, `state["history"]` will store `DialogTurn` objects that preserve the full structured context.

```python
class DialogTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str  # The speech
    metadata: PatientResponse | None = None  # Full structured data for assistant turns
    timestamp: float
```

### Metadata Collection
*   The system will aggregate the `metadata` from all turns.
*   **Post-Session**: When the simulation ends (max turns or button), save a `simulation_report.json` containing:
    *   Full structured transcript.
    *   Emotional trajectory (plot of emotional state over turns).
    *   Compliance success rate.

## 3. Robust Flow Control

### Strict Turn-Taking
*   **Mechanism**: The agent will explicitly signal "End of Turn".
*   **Interruptibility**: The "Slow Loop" (Phase 2) will monitor the `inner_thoughts` for signals to interrupt, but `speech` generation will be atomic (unless barge-in occurs).

### Role Play Limits
*   **Time Limit**: 1 Minute (approx 6-8 turns).
*   **Turn Limit**: Hard cap at 10 turns.
*   **End State**: Agent generates a `"conversation_status": "finished"` flag in metadata to gracefully close the socket.

## 4. Work Items

1.  [ ] **Schema Definition**: Create `unmute/schemas_phase3.py`.
2.  [ ] **Mistral Client Update**: Implement JSON/Pydantic validation in `chat_stream`.
3.  [ ] **Handler Refactor**:
    *   Update `_generate_response_task` to parse JSON.
    *   Split `speech` for TTS vs `inner_thoughts` for UI/Logs.
4.  [ ] **Frontend Update**:
    *   (Optional) internal-debug-view to show "Inner Thoughts".
5.  [ ] **Audio Debugging**:
    *   Strict decoder check (Sample Rate Mismatch is likely culprit).
