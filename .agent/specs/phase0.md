# Phase 0 Specification: Foundation & Data Structures

**Goal**: Initialize the core data structures and configuration for the "EmpathyAI" Simulator. This phase establishes the strict typing and prompted constraints required for the LangGraph agent.

**Source of Truth**: Derived from `mission.md`.

## 1. Environment & Package Structure
*   **Package Name**: `unmute`
*   **Dependency Manager**: `uv` (Already initialized: `pyproject.toml` and `uv.lock` exist).
*   **New Files**:
    *   `unmute/schemas.py`: Contains all Pydantic models.
    *   `unmute/prompts.py`: Contains system prompts and template strings.

## 2. Data Models (`unmute/schemas.py`)

Implement the following Pydantic models. Ensure all imports (`from pydantic import BaseModel, Field`) are present.

### A. Context & Configuration
```python
from typing import Literal, List
from pydantic import BaseModel, Field

class PatientProfile(BaseModel):
    name: str = "Alex"
    age: int = 16
    condition: str = "Type 1 Diabetes"
    # Emotional Archetype determines how the agent reacts
    emotional_archetype: Literal["Hysteric", "Stoic", "Denial", "Inquisitive"]
    knowledge_level: str = "No prior knowledge, thinks it's just the flu"

class SimulationConfig(BaseModel):
    profile: PatientProfile
    difficulty: int = Field(1, ge=1, le=5)
    # Hidden instruction injected into the system prompt
    secret_objective: str = "Refuse to believe the diagnosis until the doctor uses a specific empathy technique."
```

### B. Conversation State
```python
class DialogTurn(BaseModel):
    speaker: Literal["Doctor", "Patient"]
    text: str
    sentiment: str  # Analyzed sentiment of this specific turn
    interrupted: bool = False  # Did the doctor interrupt the patient?
    timestamp: float

class SimulationState(BaseModel):
    turns: List[DialogTurn] = []
    is_active: bool = True
    emotional_state: str = "Neutral"  # Updates dynamically as conversation progresses
```

### C. Assessment & Evaluation
```python
class AssessmentScore(BaseModel):
    empathy_score: int  # 1-100
    clarity_score: int  # 1-100
    pacing_score: int  # 1-100
    critical_feedback: List[str]  # e.g., "You interrupted the patient when they were crying."
    successful_moments: List[str]  # e.g., "Good use of silence."
```

## 3. Prompts (`unmute/prompts.py`)

Create a dedicated file for prompt management to avoid hardcoding strings in logic files.

### A. Patient System Prompt
Should accept `PatientProfile` fields as format arguments.

```python
PATIENT_SYSTEM_PROMPT = """
You are {name}, a {age}-year-old. You are currently in a hospital bed. 
Condition: {condition}.
Knowledge: {knowledge_level}.

You do NOT know you have diabetes initially. 
Current Emotional State: {emotional_state}.

Rules:
1. If the doctor uses medical jargon, get confused or angry.
2. If the doctor is empathetic, soften up.
3. Keep responses under 2 sentences to allow for conversation flow.
4. Secret Objective: {secret_objective}
"""
```

### B. Evaluator Prompt
For Phase 3 reuse.

```python
EVALUATOR_SYSTEM_PROMPT = """
You are an expert medical communication coach. 
Evaluate the following conversation based on the 'SPIKES' protocol for breaking bad news.
Return a valid JSON object matching the AssessmentScore schema.
"""
```

## 4. Verification Plan
1.  **Static Analysis**: Run `uv run mypy unmute/schemas.py` to verify type integrity.
2.  **Import Test**: Run `python -c "from unmute.schemas import SimulationState; print('Schemas imported successfully')"` to ensure no syntax errors.
