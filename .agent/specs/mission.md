This is a great concept for a hackathon. To make this work with **Antigravity IDE (Vibe Coding)**, the spec needs to be strictly typed and architecturally sound before you start generating code.

Here is the refined **Mission Document**. I have tightened the architecture to address the "latency bottleneck" and the "interruption" logic, which are the hardest parts of Voice AI.

### Architectural Refinements (Read this first)

1.  **The "Sandwich" Problem:** `STT -> LLM -> TTS` is too slow for a realistic "interruptible" conversation.
    *   **Solution:** We will use **LangGraph** to manage the state. We will implement a "Streaming" approach where the Agent emits a "filler" or "reaction" token immediately (low latency) while generating the full response (high latency).
2.  **Infrastructure:** I will assume "Gradium" is the UI wrapper (likely based on Gradio). We will use **FastAPI** internally for the backend logic if the repo supports it, otherwise, we will build strictly typed Python functions that Gradium calls.
3.  **Simulation State:** We need a JSON/Pydantic schema to track the simulation so the Phase 3 analysis has structured data to analyze, not just raw text.

---

# 🚀 Mission Document: EmpathyAI - Diabetes T1 Diagnosis Trainer

**Goal:** Build a Voice AI simulator where doctors practice announcing a Type 1 Diabetes diagnosis to a simulated teenage patient.
**Stack:** Python (uv), Mistral API, LangGraph, Gradium (UI/API), TTS/STT.

## 1. System Architecture & Data Models

We utilize **Spec-Driven Development**. The following Pydantic models define the "Truth" of our application.

### A. The Context (Page 1)
```python
from pydantic import BaseModel, Field
from typing import List, Literal

class PatientProfile(BaseModel):
    name: str = "Alex"
    age: int = 16
    condition: str = "Type 1 Diabetes"
    # Emotional Archetype determines how the agent reacts (e.g., "Denial", "Hostile", "Stoic")
    emotional_archetype: Literal["Hysteric", "Stoic", "Denial", "Inquisitive"]
    knowledge_level: str = "No prior knowledge, thinks it's just the flu"
    
class SimulationConfig(BaseModel):
    profile: PatientProfile
    difficulty: int = Field(1, ge=1, le=5)
    # Hidden instruction injected into the system prompt
    secret_objective: str = "Refuse to believe the diagnosis until the doctor uses a specific empathy technique."
```

### B. The Conversation Loop (Page 2)
```python
class DialogTurn(BaseModel):
    speaker: Literal["Doctor", "Patient"]
    text: str
    sentiment: str # Analyzed sentiment of this specific turn
    interrupted: bool = False # Did the doctor interrupt the patient?
    timestamp: float

class SimulationState(BaseModel):
    turns: List[DialogTurn] = []
    is_active: bool = True
    emotional_state: str = "Neutral" # Updates dynamically as conversation progresses
```

### C. The Assessment (Page 3)
```python
class AssessmentScore(BaseModel):
    empathy_score: int # 1-100
    clarity_score: int # 1-100
    pacing_score: int # 1-100
    critical_feedback: List[str] # "You interrupted the patient when they were crying."
    successful_moments: List[str] # "Good use of silence."
```

---

## 2. Agent Logic (LangGraph Spec)

The Agent is not just a chatbot; it is a **State Machine**.

**Graph Nodes:**
1.  **`Listener`**: Receives STT input. Uses VAD (Voice Activity Detection) logic (if available in Gradium) or end-of-speech detection.
2.  **`SentimentAnalyzer` (Fast Loop)**: Instantly detects if the doctor is aggressive or soothing. Updates `SimulationState.emotional_state`.
3.  **`Responder` (Slow Loop)**: Generates the actual text response based on the `emotional_archetype`.
4.  **`Interrupter` (Async Event)**: If the doctor speaks *while* the agent is outputting TTS, this event kills the TTS stream and loops back to `Listener`.

**Mistral System Prompt Strategy:**
> "You are Alex, a 16-year-old. You are currently in a hospital bed. You do NOT know you have diabetes. You feel [emotional_state]. If the doctor uses medical jargon, get confused/angry. If the doctor is empathetic, soften up. Keep responses under 2 sentences to allow for conversation flow."

---

## 3. Phase Implementation Plan

### Phase 0: Setup & Specs (The Foundation)
*   **Goal:** Initialize the repo and create the data structures.
*   **Action:**
    1.  Initialize `uv` environment.
    2.  Create `schemas.py` with the Pydantic models defined above.
    3.  Create `prompts.py` containing the raw Mistral system prompts for the Patient and the Evaluator.

### Phase 1: The Brain (Backend & Logic)
*   **Goal:** Build the LangGraph agent that simulates the teen.
*   **Action:**
    1.  **Profile Generator:** Implement a function that takes inputs (Age, Mood) and generates the full `PatientProfile`.
    2.  **LangGraph Workflow:**
        *   Define the State Graph.
        *   Implement `process_doctor_input(text: str, state: SimulationState)` -> Returns `PatientResponse`.
    3.  **Testing:** Create a `test_agent.py` script that runs a text-only simulation in the terminal to verify the "Teenager" personality works before adding voice.

### Phase 2: The Voice Loop (Integration)
*   **Goal:** Connect the Brain to the Mouth/Ears (Gradium).
*   **Action:**
    1.  **UI Page 1:** Form to select "Patient Mood" (Dropdown) -> Instantiates `SimulationConfig`.
    2.  **UI Page 2:** The Audio Interface.
        *   *Logic:* User speaks -> Gradium STT -> Text -> LangGraph -> Text -> Gradium TTS -> Audio.
    3.  **Latency Optimization:**
        *   Implement **Streaming Response**: The moment Mistral generates the first sentence, send it to TTS. Do not wait for the full paragraph.
        *   *Vibe Check:* If the simulator feels slow, add "Audio Fillers" (e.g., recorded "Um...", "Uh-huh", "Okay...") that play *immediately* while Mistral thinks.

### Phase 3: Analysis & Refinement (The Closer)
*   **Goal:** The Report Card.
*   **Action:**
    1.  **UI Page 3:** Display the results.
    2.  **Analyzer Agent:** Create a separate Mistral chain that takes `SimulationState.turns` and compares it against `medical_guidelines.txt`.
    3.  **Output:** Render the `AssessmentScore` as a visual card (Speedometer for Empathy, List for Feedback).

---

## 4. Critical Constraints (The "Don't Break" List)

1.  **Latency:** The total time from Doctor finishing a sentence to Patient starting to speak must be under **3 seconds** (ideally 1.5s). Use Mistral-7B-Instruct (or smallest viable model) for speed, or Mistral Large via API with streaming.
2.  **State Preservation:** The context (Patient Name, Disease info) must persist across the UI reload between Page 1 and Page 2. Use `gr.State` or equivalent.
3.  **Error Handling:** If STT fails (silence), the agent should not crash. It should output a non-verbal cue (e.g., *Patient stares blankly*).