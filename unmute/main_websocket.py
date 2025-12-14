import re
import asyncio
import base64
import json
import logging
import os # Added os import
from functools import cache, partial
from typing import Annotated

import numpy as np
import requests
import sphn
from mistralai import Mistral # Added for report endpoint
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse # Added HTMLResponse
from fastapi.websockets import WebSocketState
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, computed_field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.exceptions import (
    MissingServiceAtCapacity,
    MissingServiceTimeout,
    WebSocketClosedError,
    make_ora_error,
)
from unmute.gradium_constants import (
    GRADIUM_BASE_URL,
    LLM_API_KEY,
    LLM_BASE_URL,
    MAX_VOICE_FILE_SIZE_MB,
    STT_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    VOICE_CLONING_SERVER,
)
from unmute.utils import (
    async_ttl_cached,
    AdditionalOutputs,
    CloseStream,
    audio_to_float32,
)

from unmute.timer import Stopwatch
from unmute.tts.voice_cloning import clone_voice
from unmute.tts.voices import VoiceList
from pathlib import Path
import json
from unmute.unmute_handler import UnmuteHandler

app = FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# We prefer to scale this by running more instances of the server than having a single
# server handle more. This is to avoid the GIL.
MAX_CLIENTS = 4
SEMAPHORE = asyncio.Semaphore(MAX_CLIENTS)

Instrumentator().instrument(app).expose(app)
PROFILE_ACTIVE = False
_last_profile = None
_current_profile = None

ClientEventAdapter = TypeAdapter(
    Annotated[ora.ClientEvent, Field(discriminator="type")]
)

# Allow CORS for local development
CORS_ALLOW_ORIGINS = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "You've reached the Unmute backend server."}


if PROFILE_ACTIVE:

    @app.get("/profile")
    def profile():
        if _last_profile is None:
            return HTMLResponse("<body>No last profiler saved</body>")
        else:
            return HTMLResponse(_last_profile.output_html())  # type: ignore


def _check_server_status(server_url: str, headers: dict | None = None) -> bool:
    """Check if the server is up by sending a GET request."""
    try:
        response = requests.get(
            server_url,
            timeout=2,
            headers=headers or {},
        )
        logger.info(f"Response from {server_url}: {response}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.info(f"Couldn't connect to {server_url}: {e}")
        return False


async def debug_running_tasks():
    while True:
        logger.debug(f"Running tasks: {len(asyncio.all_tasks())}")
        for task in asyncio.all_tasks():
            logger.debug(f"  Task: {task.get_name()} - {task.get_coro()}")
        await asyncio.sleep(5)


class HealthStatus(BaseModel):
    tts_up: bool
    stt_up: bool
    llm_up: bool
    voice_cloning_up: bool

    @computed_field
    @property
    def ok(self) -> bool:
        # TODO(laurent): hack for getting things started.
        return True
        # Note that voice cloning is not required for the server to be healthy.
        # return self.tts_up and self.stt_up and self.llm_up


@partial(async_ttl_cached, ttl_sec=0.5)
async def _get_health(
    _none: None,
):  # dummy param _none because caching function expects a single param as cache key.
    async with asyncio.TaskGroup() as tg:
        tts_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status, GRADIUM_BASE_URL.url("/api/build_info")
            )
        )
        stt_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status, GRADIUM_BASE_URL.url("/api/build_info")
            )
        )
        llm_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status,
                LLM_BASE_URL.url("/v1/models"),
                # The default vLLM server doesn't use auth, but this is needed if you
                # use OpenAI or another LLM server.
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
        )
        tts_up_res = await tts_up
        stt_up_res = await stt_up
        llm_up_res = await llm_up

    return HealthStatus(
        tts_up=tts_up_res,
        stt_up=stt_up_res,
        llm_up=llm_up_res,
        voice_cloning_up=False,
    )


@app.get("/v1/health")
async def get_health():
    health = await _get_health(None)
    mt.HEALTH_OK.observe(health.ok)
    return health


@app.get("/v1/voices")
@cache
def voices():
    voice_list = VoiceList()
    # Note that `voice.good` is bool | None, here we really take only True values.
    good_voices = [
        voice.model_dump(exclude={"comment"})
        for voice in voice_list.voices
        if voice.good
    ]
    return good_voices


class LimitUploadSizeForPath(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_upload_size: int, path: str) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size
        self.path = path

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "POST" and request.url.path == self.path:
            if "content-length" not in request.headers:
                return Response(status_code=status.HTTP_411_LENGTH_REQUIRED)

            content_length = int(request.headers["content-length"])
            if content_length > self.max_upload_size:
                return Response(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        return await call_next(request)


app.add_middleware(
    LimitUploadSizeForPath,
    max_upload_size=MAX_VOICE_FILE_SIZE_MB * 1024 * 1024,
    path="/v1/voices",
)


@app.post("/v1/voices")
async def post_voices(file: UploadFile):
    """Upload a voice list file.

    Make sure the maximum file size is configured in uvicorn.
    """
    name = clone_voice(file.file.read())
    return {"name": name}


@app.get("/v1/scenarios")
async def get_scenarios():
    try:
        path = Path(__file__).parent.parent / "states" / "patient.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            scenarios = data.get("patient_llm_config", {}).get("scenarios", [])
            return {"scenarios": scenarios}
    except Exception as e:
        logger.error(f"Failed to load scenarios: {e}")
        return {"scenarios": [], "error": str(e)}

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

@app.post("/v1/report")
async def generate_report(request: TranscriptRequest):
    transcript = request.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    # Build prompt with embedded transcript
    prompt = SYSTEM_PROMPT.format(transcript=transcript)

    try:
        # Use MISTRAL_API_KEY directly as LLM_API_KEY might be for proxy
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
             raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set")
             
        client = Mistral(api_key=api_key)

        # Chat completion request - using async to avoid blocking
        response = await client.chat.complete_async(
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
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/report/html", response_class=HTMLResponse)
async def generate_report_html(request: TranscriptRequest):
    # Reuse the JSON generation logic
    try:
        report_data = await generate_report(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("HTML Report generation failed during JSON step")
        raise HTTPException(status_code=500, detail=str(e))

    # HTML Template
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Unmute Evaluation Report</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #111; color: #f0f0f0; }}
            .score-high {{ color: #4ade80; background: rgba(74, 222, 128, 0.1); }}
            .score-med {{ color: #facc15; background: rgba(250, 204, 21, 0.1); }}
            .score-low {{ color: #f87171; background: rgba(248, 113, 113, 0.1); }}
        </style>
    </head>
    <body class="p-8 max-w-4xl mx-auto">
        <header class="mb-12 border-b border-gray-800 pb-8">
            <h1 class="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-red-500 mb-2">
                Evaluation Report
            </h1>
            <p class="text-gray-400">Generated by Unmute AI</p>
        </header>

        <section class="mb-12">
            <h2 class="text-2xl font-bold mb-6 flex items-center gap-2">
                <span>📝</span> Key Recommendations
            </h2>
            <div class="grid gap-6">
                <div class="p-6 rounded-xl border border-green-900/50 bg-green-900/10">
                    <h3 class="font-bold text-green-400 mb-2">Positive Aspects</h3>
                    <p class="text-gray-300 leading-relaxed">{report_data['recommendations']['positive_aspects']}</p>
                </div>
                <div class="p-6 rounded-xl border border-red-900/50 bg-red-900/10">
                    <h3 class="font-bold text-red-400 mb-2">Areas to Improve</h3>
                    <p class="text-gray-300 leading-relaxed">{report_data['recommendations']['areas_to_improve']}</p>
                </div>
                <div class="p-6 rounded-xl border border-blue-900/50 bg-blue-900/10">
                    <h3 class="font-bold text-blue-400 mb-2">Patient's Emotional Response</h3>
                    <p class="text-gray-300 leading-relaxed">{report_data['recommendations']['patient_emotional_response']}</p>
                </div>
            </div>
        </section>

        <section class="mb-12">
            <h2 class="text-2xl font-bold mb-6">Detailed Analysis</h2>
            <div class="space-y-6">
                {_render_category_html("Diagnosis Delivery", report_data['evaluation']['diagnosis_delivery'])}
                {_render_category_html("Emotional Acknowledgement", report_data['evaluation']['emotional_acknowledgement'])}
                {_render_category_html("Language Accessibility", report_data['evaluation']['language_accessibility'])}
                {_render_category_html("Emotional Responsiveness", report_data['evaluation']['emotional_responsiveness'])}
                {_render_category_html("Interaction Balance", report_data['evaluation']['interaction_balance'])}
                {_render_category_html("Identity & Future Preservation", report_data['evaluation']['identity_future_preservation'])}
            </div>
        </section>

        <section class="mt-16 pt-8 border-t border-gray-800 opacity-60">
            <h3 class="text-lg font-bold mb-4">Transcript Reference</h3>
            <pre class="whitespace-pre-wrap bg-black p-6 rounded-lg text-sm font-mono text-gray-400 border border-gray-800 overflow-x-auto">{report_data['transcript']}</pre>
        </section>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def _render_category_html(title, data):
    score = data['score']
    score_class = "score-high" if score >= 4 else "score-med" if score >= 3 else "score-low"
    
    return f"""
    <div class="bg-white/5 p-6 rounded-xl border border-white/10">
        <div class="flex justify-between items-center mb-4">
            <h3 class="text-xl font-bold text-white">{title}</h3>
            <span class="px-4 py-1 rounded-full font-bold {score_class}">
                Score: {score}/5
            </span>
        </div>
        <p class="text-gray-300 mb-4">{data['justification']}</p>
        <div class="grid md:grid-cols-2 gap-6 text-sm">
            <div>
                <span class="text-red-400 font-semibold block mb-1">Missed Opportunity:</span>
                <span class="text-gray-400">{data['missed_opportunity']}</span>
            </div>
            <div>
                <span class="text-blue-400 font-semibold block mb-1">Feedback:</span>
                <span class="text-gray-400">{data['feedback']}</span>
            </div>
        </div>
    </div>
    """

@app.websocket("/v1/realtime")
async def websocket_route(websocket: WebSocket):
    global _last_profile, _current_profile
    mt.SESSIONS.inc()
    mt.ACTIVE_SESSIONS.inc()
    session_watch = Stopwatch()
    if PROFILE_ACTIVE and _current_profile is None:
        from pyinstrument import Profiler

        logger.info("Profiler started.")
        _current_profile = Profiler(interval=0.0001, async_mode="disabled")
        import inspect

        frame = inspect.currentframe()
        while frame is not None and frame.f_back:
            frame = frame.f_back
        _current_profile.start(caller_frame=frame)

    async with SEMAPHORE:
        try:
            # The `subprotocol` argument is important because the client specifies what
            # protocol(s) it supports and OpenAI uses "realtime" as the value. If we
            # don't set this, the client will think this is not the right endpoint and
            # will not connect.
            await websocket.accept(subprotocol="realtime")

            handler = UnmuteHandler()
            async with handler:
                await handler.start_up()
                await _run_route(websocket, handler)

        except Exception as exc:
            await _report_websocket_exception(websocket, exc)
        finally:
            if _current_profile is not None:
                _current_profile.stop()
                logger.info("Profiler saved.")
                _last_profile = _current_profile
                _current_profile = None

            mt.ACTIVE_SESSIONS.dec()
            mt.SESSION_DURATION.observe(session_watch.time())


async def _report_websocket_exception(websocket: WebSocket, exc: Exception):
    if isinstance(exc, ExceptionGroup):
        exceptions = exc.exceptions
    else:
        exceptions = [exc]

    error_message = None

    for exc in exceptions:
        if isinstance(exc, (MissingServiceAtCapacity)):
            mt.FATAL_SERVICE_MISSES.inc()
            error_message = (
                f"Too many people are connected to service '{exc.service}'. "
                "Please try again later."
            )
        elif isinstance(exc, MissingServiceTimeout):
            mt.FATAL_SERVICE_MISSES.inc()
            error_message = (
                f"Service '{exc.service}' timed out. Please try again later."
            )
        elif isinstance(exc, WebSocketClosedError):
            logger.debug("Websocket was closed.")
        else:
            logger.exception("Unexpected error: %r", exc)
            mt.HARD_ERRORS.inc()
            error_message = f"Internal server error: {exc}"

    if error_message is not None:
        mt.FORCE_DISCONNECTS.inc()

        try:
            await websocket.send_text(
                make_ora_error(type="fatal", message=error_message).model_dump_json()
            )
        except WebSocketDisconnect:
            logger.warning("Failed to send error message due to disconnect.")

        try:
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason=error_message,
            )
        except RuntimeError:
            logger.warning("Socket already closed.")


async def _run_route(websocket: WebSocket, handler: UnmuteHandler):
    health = await get_health()
    if not health.ok:
        logger.info("Health check failed, closing WebSocket connection.")
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason=f"Server is not healthy: {health}",
        )
        return

    emit_queue: asyncio.Queue[ora.ServerEvent] = asyncio.Queue()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                receive_loop(websocket, handler, emit_queue), name="receive_loop()"
            )
            tg.create_task(
                emit_loop(websocket, handler, emit_queue), name="emit_loop()"
            )
            tg.create_task(handler.quest_manager.wait(), name="quest_manager.wait()")
            tg.create_task(debug_running_tasks(), name="debug_running_tasks()")
    finally:
        await handler.cleanup()
        logger.info("websocket_route() finished")


async def receive_loop(
    websocket: WebSocket,
    handler: UnmuteHandler,
    emit_queue: asyncio.Queue[ora.ServerEvent],
):
    """Receive messages from the WebSocket.

    Can decide to send messages via `emit_queue`.
    """
    opus_reader = sphn.OpusStreamReader(STT_SAMPLE_RATE)
    wait_for_first_opus = True
    while True:
        try:
            message_raw = await websocket.receive_text()
        except WebSocketDisconnect as e:
            logger.info(
                "receive_loop() stopped because WebSocket disconnected: "
                f"{e.code=} {e.reason=}"
            )
            raise WebSocketClosedError() from e
        except RuntimeError as e:
            # This is expected when the client disconnects
            if "WebSocket is not connected" not in str(e):
                raise  # re-raise unexpected errors

            logger.info("receive_loop() stopped because WebSocket disconnected.")
            raise WebSocketClosedError() from e

        try:
            message: ora.ClientEvent = ClientEventAdapter.validate_json(message_raw)
        except json.JSONDecodeError as e:
            await emit_queue.put(
                ora.Error(
                    error=ora.ErrorDetails(
                        type="invalid_request_error",
                        message=f"Invalid JSON: {e}",
                    )
                )
            )
            continue
        except ValidationError as e:
            await emit_queue.put(
                ora.Error(
                    error=ora.ErrorDetails(
                        type="invalid_request_error",
                        message="Invalid message",
                        details=json.loads(e.json()),
                    )
                )
            )
            continue

        message_to_record = message

        if isinstance(message, ora.InputAudioBufferAppend):
            opus_bytes = base64.b64decode(message.audio)
            if wait_for_first_opus:
                # Somehow the UI is sending us potentially old messages from a previous
                # connection on reconnect, so that we might get some old OGG packets,
                # waiting for the bit set for first packet to feed to the decoder.
                if opus_bytes[5] & 2:
                    wait_for_first_opus = False
                else:
                    continue
            pcm = await asyncio.to_thread(opus_reader.append_bytes, opus_bytes)

            message_to_record = ora.UnmuteInputAudioBufferAppendAnonymized(
                number_of_samples=pcm.size,
            )

            if pcm.size:
                await handler.receive((STT_SAMPLE_RATE, pcm[np.newaxis, :]))
        elif isinstance(message, ora.SessionUpdate):
            await handler.update_session(message.session)
            await emit_queue.put(ora.SessionUpdated(session=message.session))

        elif isinstance(message, ora.UnmuteAdditionalOutputs):
            # Don't record this: it's a debugging message and can be verbose. Anything
            # important to store should be in the other event types.
            message_to_record = None

        else:
            logger.info("Ignoring message:", str(message)[:100])

        if message_to_record is not None and handler.recorder is not None:
            await handler.recorder.add_event("client", message_to_record)


class EmitDebugLogger:
    def __init__(self):
        self.last_emitted_n = 0
        self.last_emitted_type = ""

    def on_emit(self, to_emit: ora.ServerEvent):
        if self.last_emitted_type == to_emit.type:
            self.last_emitted_n += 1
        else:
            self.last_emitted_n = 1
            self.last_emitted_type = to_emit.type

        if self.last_emitted_n == 1:
            logger.debug(f"Emitting: {to_emit.type}")
        else:
            logger.debug(f"Emitting ({self.last_emitted_n}): {self.last_emitted_type}")


async def emit_loop(
    websocket: WebSocket,
    handler: UnmuteHandler,
    emit_queue: asyncio.Queue[ora.ServerEvent],
):
    """Send messages to the WebSocket."""
    emit_debug_logger = EmitDebugLogger()

    opus_writer = sphn.OpusStreamWriter(TTS_SAMPLE_RATE)

    while True:
        if (
            websocket.application_state == WebSocketState.DISCONNECTED
            or websocket.client_state == WebSocketState.DISCONNECTED
        ):
            logger.info("emit_loop() stopped because WebSocket disconnected")
            raise WebSocketClosedError()

        try:
            to_emit = emit_queue.get_nowait()
        except asyncio.QueueEmpty:
            emitted_by_handler = await handler.emit()

            if emitted_by_handler is None:
                continue
            elif isinstance(emitted_by_handler, AdditionalOutputs):
                assert len(emitted_by_handler.args) == 1
                to_emit = ora.UnmuteAdditionalOutputs(
                    args=emitted_by_handler.args[0],
                )
            elif isinstance(emitted_by_handler, CloseStream):
                # Close here explicitly so that the receive loop stops too
                await websocket.close()
                break
            elif isinstance(emitted_by_handler, ora.ServerEvent):
                to_emit = emitted_by_handler
            else:
                _sr, audio = emitted_by_handler
                audio = audio_to_float32(audio)
                opus_bytes = await asyncio.to_thread(opus_writer.append_pcm, audio)
                # Due to buffering/chunking, Opus doesn't necessarily output something on every PCM added
                if opus_bytes:
                    to_emit = ora.ResponseAudioDelta(
                        delta=base64.b64encode(opus_bytes).decode("utf-8"),
                    )
                else:
                    continue

        emit_debug_logger.on_emit(to_emit)

        if handler.recorder is not None:
            await handler.recorder.add_event("server", to_emit)

        try:
            await websocket.send_text(to_emit.model_dump_json())
        except (WebSocketDisconnect, RuntimeError) as e:
            if isinstance(e, RuntimeError):
                if "Unexpected ASGI message 'websocket.send'" in str(e):
                    # This is expected when the client disconnects
                    message = f"emit_loop() stopped because WebSocket disconnected: {e}"
                else:
                    raise
            else:
                message = (
                    "emit_loop() stopped because WebSocket disconnected: "
                    f"{e.code=} {e.reason=}"
                )

            logger.info(message)
            raise WebSocketClosedError() from e


def _cors_headers_for_error(request: Request):
    origin = request.headers.get("origin")
    allow_origin = origin if origin in CORS_ALLOW_ORIGINS else None
    headers = {"Access-Control-Allow-Credentials": "true"}
    if allow_origin:
        headers["Access-Control-Allow-Origin"] = allow_origin

    return headers


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # We need this so that CORS header are added even when the route raises an
    # exception. Otherwise you get a confusing CORS error even if the issue is totally
    # unrelated.
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=_cors_headers_for_error(request),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # We need this so that CORS header are added even when the route raises an
    # exception. Otherwise you get a confusing CORS error even if the issue is totally
    # unrelated.
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=_cors_headers_for_error(request),
    )


if __name__ == "__main__":
    import sys

    print(f"Run this via:\nfastapi dev {sys.argv[0]}")
    exit(1)
