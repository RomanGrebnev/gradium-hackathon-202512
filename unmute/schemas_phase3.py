from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

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
    conversation_status: Literal["continue", "finished"] = Field(
        default="continue",
        description="Set to 'finished' if the conversation has reached a natural conclusion or the turn limit is reached."
    )
    metadata: Dict[str, Any] = Field(
        default={},
        description="Any additional metric to track (e.g., 'doctor_empathy_perceived': 1-10)."
    )

class StructuredDialogTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str  # The raw content (speech for assistant, input for user)
    structured_response: PatientResponse | None = None  # Full structured data for assistant turns
    timestamp: float
