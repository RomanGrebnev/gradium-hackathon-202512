import asyncio
import base64
import numpy as np
import json
import urllib.parse
from logging import getLogger
from typing import Annotated, AsyncIterator, Callable, Literal, Union

import websockets
from pydantic import BaseModel, Field, TypeAdapter, field_validator

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.gradium_constants import (
    FRAME_TIME_SEC,
    HEADERS,
    TTS_SAMPLE_RATE,
    TEXT_TO_SPEECH_PATH,
    GRADIUM_BASE_URL,
    BaseUrl,
)
from unmute.recorder import Recorder
from unmute.timer import Stopwatch
from unmute.tts.realtime_queue import RealtimeQueue
from unmute.utils import ServiceWithStartup, WebsocketState

logger = getLogger(__name__)


class TTSClientTextMessage(BaseModel):
    """Message sent to the TTS server saying we to turn this text into speech."""

    type: Literal["text"] = "text"
    text: str


class TTSClientEosMessage(BaseModel):
    """Message sent to the TTS server saying we are done sending text."""

    type: Literal["end_of_stream"] = "end_of_stream"


TTSClientMessage = Annotated[
    Union[TTSClientTextMessage, TTSClientEosMessage],
    Field(discriminator="type"),
]


class TTSTextMessage(BaseModel):
    type: Literal["text"]
    text: str
    start_s: float
    stop_s: float


class TTSAudioMessage(BaseModel):
    type: Literal["audio"]
    audio: bytes


class TTSErrorMessage(BaseModel):
    type: Literal["error"]
    message: str


class TTSReadyMessage(BaseModel):
    type: Literal["ready"]


class TTSEosMessage(BaseModel):
    type: Literal["end_of_stream"]


TTSMessage = Annotated[
    Union[
        TTSTextMessage, TTSAudioMessage, TTSErrorMessage, TTSReadyMessage, TTSEosMessage
    ],
    Field(discriminator="type"),
]
TTSMessageAdapter = TypeAdapter(TTSMessage)


class QueueAudio(BaseModel):
    type: Literal["audio"] = "audio"
    pcm: np.ndarray

    @field_validator("pcm", mode="before")
    def validate_pcm(cls, v: np.ndarray):
        if v.dtype != np.float32:
            raise ValueError("pcm must be float32")
        if v.ndim != 1:
            raise ValueError("pcm must be 1-dimensional")
        return v

    class Config:
        arbitrary_types_allowed = True


class QueueText(BaseModel):
    type: Literal["text"] = "text"
    text: str


QueueMessage = Annotated[Union[QueueAudio, QueueText], Field(discriminator="type")]
QueueMessageAdapter = TypeAdapter(QueueMessage)


def url_escape(value: object) -> str:
    return urllib.parse.quote(str(value), safe="")


# Only release the audio such that it's AUDIO_BUFFER_SEC ahead of real time.
# If the value it's too low, it might cause stuttering.
# If it's too high, it's difficult to control the synchronization of the text and the
# audio, because that's controlled by emit() and WebRTC. Note that some
# desynchronization can still occur if the TTS is less than real-time, because WebRTC
# will decide to do some buffering of the audio on the fly.
AUDIO_BUFFER_SEC = FRAME_TIME_SEC * 4


def prepare_text_for_tts(text: str) -> str:
    text = text.strip()

    unpronounceable_chars = "*_`"
    for char in unpronounceable_chars:
        text = text.replace(char, "")

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace(" : ", " ")

    return text


class TtsStreamingQuery(BaseModel):
    # See moshi-rs/moshi-server/
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    format: str = "PcmMessagePack"
    voice: str | None = None
    voices: list[str] | None = None
    max_seq_len: int | None = None
    cfg_alpha: float | None = None
    auth_id: str | None = None

    def to_url_params(self) -> str:
        params = self.model_dump()
        return "?" + "&".join(
            f"{key}={url_escape(value)}"
            for key, value in params.items()
            if value is not None
        )


class TextToSpeech(ServiceWithStartup):
    def __init__(
        self,
        base_url: BaseUrl = GRADIUM_BASE_URL,
        # For TTS, we do internal queuing, so we pass in the recorder to be able to
        # record the true time of the messages.
        recorder: Recorder | None = None,
        get_time: Callable[[], float] | None = None,
        voice: str | None = None,
    ):
        self.base_url = base_url
        self.recorder = recorder
        self.websocket: websockets.ClientConnection | None = None

        self.time_since_first_text_sent = Stopwatch(autostart=False)
        self.waiting_first_audio: bool = True
        # Number of samples received from the TTS server
        self.received_samples = 0
        # Number of samples that we passed on after waiting for the correct time
        self.received_samples_yielded = 0

        self.voice = voice
        self.query = TtsStreamingQuery(
            voice=self.voice
            # Don't pass in custom voices as a query parameter, we set it later using
            # a message
            if (self.voice and not self.voice.startswith("custom:"))
            else None,
            cfg_alpha=1.5,
        )

        # self.query_parameters = f"?voice={self.voice}&cfg_alpha=2&format=PcmMessagePack"
        self.text_output_queue = RealtimeQueue(get_time=get_time)

        self.shutdown_lock = asyncio.Lock()
        self.shutdown_complete = asyncio.Event()

    def state(self) -> WebsocketState:
        if not self.websocket:
            return "not_created"
        else:
            d: dict[websockets.protocol.State, WebsocketState] = {
                websockets.protocol.State.CONNECTING: "connecting",
                websockets.protocol.State.OPEN: "connected",
                websockets.protocol.State.CLOSING: "closing",
                websockets.protocol.State.CLOSED: "closed",
            }
            return d[self.websocket.state]

    async def send(self, message: str | TTSClientMessage) -> None:
        """Send a message to the TTS server.

        Note that raw strings will be preprocessed to remove unpronounceable characters
        etc., but a TTSClientTextMessage will send the text as-is.
        """
        if isinstance(message, str):
            message = TTSClientTextMessage(
                type="text", text=prepare_text_for_tts(message)
            )

        if self.shutdown_lock.locked():
            logger.warning("Can't send - TTS shutting down")
        elif not self.websocket:
            logger.warning("Can't send - TTS websocket not connected")
        else:
            if isinstance(message, TTSClientTextMessage):
                if message.text == "":
                    return  # Don't send empty messages

                mt.TTS_SENT_FRAMES.inc()
                self.time_since_first_text_sent.start_if_not_started()

            await self.websocket.send(json.dumps(message.model_dump()))

    async def start_up(self):
        url = self.base_url.ws_url(TEXT_TO_SPEECH_PATH)
        # TODO(laurent): use the following in the settup message self.query.to_url_params()
        logger.info(f"Connecting to TTS: {url}")
        self.websocket = await websockets.connect(
            url,
            additional_headers=HEADERS,
        )
        logger.debug("Connected to TTS")

        try:
            # TODO(laurent): voice etc.
            setup = {
                "type": "setup",
                "model_name": "default",
                "voice": str(self.voice),
                "output_format": "pcm",
            }
            await self.websocket.send(json.dumps(setup))
            message = await self.websocket.recv(decode=False)
            message_dict = json.loads(message)
            message = TTSMessageAdapter.validate_python(message_dict)
            if isinstance(message, TTSReadyMessage):
                return
            else:
                logger.warning(
                    f"Received unexpected message type from {self.base_url}, {message.type}"
                )
        except Exception as e:
            logger.error(f"Error during TTS startup: {repr(e)}")
            # Make sure we don't leave a dangling websocket connection
            await self.websocket.close()
            self.websocket = None
            raise

        raise AssertionError("Not supposed to happen.")

    async def shutdown(self):
        async with self.shutdown_lock:
            if self.shutdown_complete.is_set():
                return
            mt.TTS_ACTIVE_SESSIONS.dec()
            mt.TTS_AUDIO_DURATION.observe(self.received_samples / TTS_SAMPLE_RATE)
            if self.time_since_first_text_sent.started:
                mt.TTS_GEN_DURATION.observe(self.time_since_first_text_sent.time())

            # Set before closing the websocket so that __aiter__ knows we're closing
            # the connection intentionally
            self.shutdown_complete.set()

            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            logger.info("TTS shutdown() finished")

    async def __aiter__(self) -> AsyncIterator[QueueMessage]:
        if self.websocket is None:
            raise RuntimeError("TTS websocket not connected")
        mt.TTS_SESSIONS.inc()
        mt.TTS_ACTIVE_SESSIONS.inc()

        output_queue: RealtimeQueue[QueueMessage] = RealtimeQueue()

        try:
            async for message_s in self.websocket:
                message_dict = json.loads(message_s)
                message: TTSMessage = TTSMessageAdapter.validate_python(message_dict)

                if isinstance(message, TTSAudioMessage):
                    # Use `yield message` if you want to to release the audio
                    # as fast as it's being generated. However, it might desynchronize
                    # the text and the audio.
                    mt.TTS_RECV_FRAMES.inc()
                    if (
                        self.waiting_first_audio
                        and self.time_since_first_text_sent.started
                    ):
                        self.waiting_first_audio = False
                        ttft = self.time_since_first_text_sent.time()
                        mt.TTS_TTFT.observe(ttft)
                        logger.info("Time to first token is %.1f ms", ttft * 1000)
                    output_queue.start_if_not_started()
                    pcm = base64.b64decode(message.audio)
                    pcm = (np.frombuffer(pcm, dtype=np.int16) / 32768.0).astype(
                        np.float32
                    )
                    output_queue.put(
                        QueueAudio(pcm=pcm),
                        self.received_samples / TTS_SAMPLE_RATE - AUDIO_BUFFER_SEC,
                    )
                    self.received_samples += len(pcm)

                    if self.recorder is not None:
                        await self.recorder.add_event(
                            "server",
                            ora.UnmuteResponseAudioDeltaReady(
                                number_of_samples=len(pcm)
                            ),
                        )

                elif isinstance(message, TTSTextMessage):
                    mt.TTS_RECV_WORDS.inc()
                    print(">>> MT", message)  # TODO(laurent): remove
                    # There are two reasons why we don't send the text messages
                    # immediately:
                    # - The text messages have timestamps "from the future" because the
                    # audio stream is delayed by 2s.
                    # - Even so, we receive the audio/text faster than real time. It
                    #   seems difficult to keep track of how much audio data has already
                    #   been streamed (.emit() eats up the inputs immediately,
                    #   apparently it has some internal buffering) so we only send the
                    #   text messages at the real time when they're actually supposed to
                    #   be displayed. Precise timing/buffering is less important here.
                    # By using stop_s instead of start_s, we ensure that anything shown
                    # has already been said, so that if there's an interruption, the
                    # chat history matches what's actually been said.
                    output_queue.put(QueueText(text=message.text), message.stop_s)

                for _, msg in output_queue.get_nowait():
                    if isinstance(msg, QueueAudio):
                        self.received_samples_yielded += len(msg.pcm)

                    yield msg

        except websockets.ConnectionClosedOK:
            pass
        except websockets.ConnectionClosedError:
            if self.shutdown_complete.is_set():
                # If we closed the websocket in shutdown(), it leads to this exception
                # (not sure why) but it's an intentional exit, so don't raise.
                pass
            else:
                raise

        # Empty the queue if the connection is closed - we're releasing the messages
        # in real time, see above.
        async for _, msg in output_queue:
            if self.shutdown_complete.is_set():
                break
            if isinstance(msg, QueueAudio):
                self.received_samples_yielded += len(msg.pcm)
            yield msg

        logger.debug("TTS __aiter__() finished")
        await self.shutdown()
