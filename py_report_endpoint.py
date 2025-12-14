from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
import re
import json
from mistralai import Mistral


# Load .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
   raise RuntimeError(
       "Please set the MISTRAL_API_KEY environment variable in .env or your shell."
   )


app = FastAPI(title="Diabetes Communication Evaluator")


class TranscriptRequest(BaseModel):
   transcript: str


SYSTEM_PROMPT = """
You are an expert evaluator of doctor-patient conversations, specifically focused on delivering sensitive diagnoses (like Type 1 Diabetes) to teenagers.


Your task is to evaluate the transcript of a conversation between a doctor and a teenage patient. Follow these rules:


1. **Language Matters**: Communicate respectfully, inclusively, empathetically, and clearly. Avoid blaming, stigmatizing, or using medical jargon the teenager cannot understand.
2. Evaluate the doctor’s performance across the following criteria, providing scores (0-5), justification, missed opportunities, and actionable feedback:


---


**Evaluation Criteria and Questions**:


1. **Diagnosis Delivery**
  - Did the doctor clearly explain the diagnosis?
  - Was it age-appropriate and understandable?
  - Was the information structured and not overwhelming?


2. **Emotional Acknowledgement**
  - Did the doctor recognize the patient's feelings?
  - Did they show empathy or concern?
  - Did they ask about the patient’s emotional state?


3. **Language Accessibility**
  - Was the language simple, clear, and understandable for a teenager?
  - Was technical jargon avoided or explained?
  - Were metaphors or analogies helpful?


4. **Emotional Responsiveness**
  - Did the doctor respond appropriately to the patient’s emotional cues?
  - Did they offer reassurance or support?


5. **Interaction Balance**
  - Was the conversation balanced in speaking turns?
  - Did the doctor encourage the patient to speak and ask questions?


6. **Identity & Future Preservation**
  - Did the doctor acknowledge the patient’s identity, individuality, or future goals?
  - Did they avoid framing the patient solely as a “patient” or “ill person”?


---


**Output Instructions:**
- Always output **valid JSON** following this structure.
- Include both the evaluation scores and detailed recommendations.
- Do not include Markdown or explanatory text outside the JSON.
- Be concise but specific.


---


Return this JSON structure (fill in all fields):


{{
 "transcript": "{transcript}",
 "evaluation": {{
     "diagnosis_delivery": {{"score": 0, "justification": "", "missed_opportunity": "", "feedback": ""}},
     "emotional_acknowledgement": {{"score": 0, "justification": "", "missed_opportunity": "", "feedback": ""}},
     "language_accessibility": {{"score": 0, "justification": "", "missed_opportunity": "", "feedback": ""}},
     "emotional_responsiveness": {{"score": 0, "justification": "", "missed_opportunity": "", "feedback": ""}},
     "interaction_balance": {{"score": 0, "justification": "", "missed_opportunity": "", "feedback": ""}},
     "identity_future_preservation": {{"score": 0, "justification": "", "missed_opportunity": "", "feedback": ""}}
 }},
 "recommendations": {{
     "positive_aspects": "",
     "areas_to_improve": "",
     "patient_emotional_response": ""
 }}
}}


Now evaluate the following transcript:
{transcript}
"""






@app.post("/evaluate")
def evaluate_conversation(request: TranscriptRequest):
   transcript = request.transcript.strip()
   if not transcript:
       raise HTTPException(status_code=400, detail="Transcript cannot be empty.")


   # Build prompt with embedded transcript
   prompt = SYSTEM_PROMPT.format(transcript=transcript)


   try:
       client = Mistral(api_key=MISTRAL_API_KEY)


       # Chat completion request
       response = client.chat.complete(
           model="mistral-small-latest",
           messages=[
               {"role": "system", "content": prompt},
               {"role": "user", "content": transcript},
           ],
           stream=False
       )


       # Extract content
       raw_output = response.choices[0].message.content


       # Strip Markdown backticks if present
       clean_output = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip(), flags=re.MULTILINE)


       try:
           result = json.loads(clean_output)
       except json.JSONDecodeError as e:
           raise HTTPException(
               status_code=500,
               detail=f"Failed to parse LLM output as JSON: {e}\nRaw output:\n{raw_output}"
           )


       return result


   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))