from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
import os
import urllib


load_dotenv()


@dataclass
class BaseUrl:
    use_https: bool
    netloc: str
    path: str

    def __str__(self) -> str:
        scheme = "https" if self.use_https else "http"
        return f"{scheme}://{self.netloc}{self.path}"

    def ws_scheme(self) -> str:
        return "wss" if self.use_https else "ws"

    def ws_url(self, path: str) -> str:
        return f"{self.ws_scheme()}://{self.netloc}{self.path}{path}"

    def url(self, path: str) -> str:
        scheme = "https" if self.use_https else "http"
        return f"{scheme}://{self.netloc}{self.path}{path}"


def from_url(url: str) -> BaseUrl:
    parsed = urllib.parse.urlparse(url)
    base_url = BaseUrl(
        use_https=parsed.scheme == "https",
        netloc=parsed.netloc,
        path=parsed.path,
    )
    return base_url


GRADIUM_API_KEY = os.environ.get("GRADIUM_API_KEY")
if GRADIUM_API_KEY is None:
    raise ValueError("GRADIUM_API_KEY environment variable is not set")
HEADERS = {"x-api-key": GRADIUM_API_KEY}

GRADIUM_BASE_URL = from_url(
    os.environ.get("GRADIUM_BASE_URL", "https://eu.api.gradium.ai/api/speech")
)

LLM_BASE_URL = from_url(os.environ.get("LLM_BASE_URL"))
LLM_MODEL = os.environ.get("LLM_MODEL")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
VOICE_CLONING_SERVER = os.environ.get(
    "KYUTAI_VOICE_CLONING_URL", "http://localhost:8092"
)
# If None, a dict-based cache will be used instead of Redis
REDIS_SERVER = os.environ.get("KYUTAI_REDIS_URL")

SPEECH_TO_TEXT_PATH = "/asr"
TEXT_TO_SPEECH_PATH = "/tts"

repo_root = Path(__file__).parents[1]
VOICE_DONATION_DIR = Path(
    os.environ.get("KYUTAI_VOICE_DONATION_DIR", repo_root / "voices" / "donation")
)

# If None, recordings will not be saved
_recordings_dir = os.environ.get("GRADIUM_RECORDINGS_DIR")
RECORDINGS_DIR = Path(_recordings_dir) if _recordings_dir else None

# Also checked on the frontend, see constant of the same name
MAX_VOICE_FILE_SIZE_MB = 4


STT_SAMPLE_RATE = 24000
TTS_SAMPLE_RATE = 24000
STT_SAMPLES_PER_FRAME = 1920
TTS_SAMPLES_PER_FRAME = 3840
FRAME_TIME_SEC = 0.08
# TODO: make it so that we can read this from the ASR server?
STT_DELAY_SEC = 0.5
