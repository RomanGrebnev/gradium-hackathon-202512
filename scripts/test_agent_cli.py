import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

# Load env vars *before* importing modules that might use them at top-level
load_dotenv()

from unmute.schemas import PatientProfile, DialogTurn
from unmute.agent.graph import build_graph

async def main():
    print("--- Unmute Agent CLI Test ---")
    
    if "MISTRAL_API_KEY" not in os.environ:
        print("Error: MISTRAL_API_KEY not found in environment.")
        return

    # Initialize Graph
    graph = build_graph()
    
    # Initial State
    state = {
        "patient_profile": PatientProfile(
            name="Martha", 
            age=65, 
            condition="Type 2 Diabetes", 
            base_instructions="You are finding it hard to stick to your diet."
        ),
        "history": [],
        "current_input": None,
        "emotional_state": "Neutral"
    }
    
    print(f"Patient: {state['patient_profile'].name}, {state['patient_profile'].age}")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You (Doctor): ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        print("...")
        
        # input
        state["current_input"] = user_input
        
        # Invoke Graph
        result = await graph.ainvoke(state)
        
        # Update local state with result
        state = result
        
        # Print Response
        last_turn = state["history"][-1]
        print(f"Agent ({state['emotional_state']}): {last_turn.content}\n")

if __name__ == "__main__":
    asyncio.run(main())
