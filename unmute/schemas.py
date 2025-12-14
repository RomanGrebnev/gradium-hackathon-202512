from datetime import datetime
from typing import Annotated, Literal, TypedDict, Union

from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    name: str = "Unknown"
    age: int = 50
    gender: str = "Female"
    condition: str = "Type 2 Diabetes"
    scenario_id: str = "default"
    base_instructions: str = ""


class DialogTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    metadata: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SimulationState(TypedDict):
    """
    State of the simulation graph.
    Inherits from TypedDict to capture all updates in the graph.
    """

    # Static Context
    patient_profile: PatientProfile

    # Dynamic Conversation History
    # We use a list of DialogTurn for structured history
    history: list[DialogTurn]

    # Current User Input (Transient)
    current_input: str | None

    # Internal emotional state of the simulated patient
    emotional_state: str  # e.g. "Neutral", "Anxious", "Defensive"

    # Streaming Output (Transient)
    # Allows the responder node to emit chunks
    response_stream: list[str] | None
