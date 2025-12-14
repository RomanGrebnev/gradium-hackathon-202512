from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from unmute.schemas import SimulationState, DialogTurn, PatientProfile
from unmute.agent.mistral import MistralClient

# Initialize Mistral Client
mistral = MistralClient()

def node_listen(state: SimulationState) -> SimulationState:
    """
    Ingests the current input and appends it to history.
    """
    current_input = state.get("current_input")
    if current_input:
        new_turn = DialogTurn(role="user", content=current_input)
        return {
            "history": [new_turn],
            "current_input": None # Clear input after processing
        }
    return {}

async def node_think(state: SimulationState) -> SimulationState:
    """
    Slow Loop Node: 'Think'
    Analyzes the situation to update emotional state or decide on interruption.
    Does NOT generate speech.
    """
    history = state.get("history", [])
    profile = state["patient_profile"]
    
    # Simple Heuristic for now (can be replaced by LLM call)
    # If the user has been talking for a long time (simulated by length of last turn inputs?), 
    # or used specific keywords.
    
    # We can eventually call a "CoT" model here.
    
    # Placeholder Logic:
    last_user_input = ""
    if history and history[-1].role == "user":
        last_user_input = history[-1].content.lower()

    updates = {}
    
    # Example: Keyword-based emotional shift
    if "stupid" in last_user_input or "idiot" in last_user_input:
        updates["emotional_state"] = "Angry"
    elif "worry" in last_user_input:
        updates["emotional_state"] = "Anxious"
        
    # Example: Interruption logic (if we had a 'patience' meter)
    # updates["should_interrupt"] = True/False

    return updates

async def node_respond(state: SimulationState) -> SimulationState:
    """
    Fast Loop Node: 'Respond'
    Generates a verbal response using Mistral.
    """
    profile = state["patient_profile"]
    history = state["history"]
    emotional_state = state["emotional_state"]
    
    # Construct Messages
    system_prompt = f"""You are a patient named {profile.name}.
    Condition: {profile.condition}.
    Current Emotion: {emotional_state}.
    
    Instructions: {profile.base_instructions}
    
    Respond naturally to the doctor.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
        
    # We will aggregate the stream here for the state update
    # In the Handler, we intercept this generation for streaming audio.
    full_response = ""
    async for chunk in mistral.chat_stream(messages):
        full_response += chunk
        
    new_turn = DialogTurn(role="assistant", content=full_response)
    return {"history": [new_turn]}


def build_graph():
    builder = StateGraph(SimulationState)
    
    builder.add_node("listen", node_listen)
    builder.add_node("think", node_think)
    builder.add_node("respond", node_respond)
    
    # Define a simple flow for now, but Handler can invoke specific nodes.
    # Standard flow: Listen -> Think -> Respond
    builder.add_edge(START, "listen")
    builder.add_edge("listen", "think")
    builder.add_edge("think", "respond")
    builder.add_edge("respond", END)
    
    return builder.compile()
