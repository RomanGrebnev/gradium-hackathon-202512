import asyncio
import logging
import numpy as np
import time
import typing as tp
from collections import defaultdict
from collections.abc import Awaitable
from functools import wraps
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
K = tp.TypeVar("K", bound=tp.Hashable)
V = tp.TypeVar("V")
S = tp.TypeVar("S", bound="ServiceWithStartup")


WebsocketState = tp.Literal[
    "not_created", "connecting", "connected", "closing", "closed"
]


def async_ttl_cached(func: tp.Callable[[K], Awaitable[V]], ttl_sec: float = 0.1):
    """Cache an async function with some TTL for the cached values."""
    cache: dict[K, tuple[float, V]] = {}
    locks: dict[K, asyncio.Lock] = defaultdict(asyncio.Lock)

    @wraps(func)
    async def cached(key: K):
        async with locks[key]:
            now = time.time()
            try:
                key_time, value = cache[key]
            except KeyError:
                pass
            else:
                if now - key_time < ttl_sec:
                    return value
            value = await func(key)
            cache[key] = (now, value)
            return value

    return cached


class ServiceWithStartup(tp.Protocol):
    async def start_up(self) -> None:
        """Initiate connection. Should raise an exception if the instance is not ready."""
        ...


class AdditionalOutputs:
    def __init__(self, *args) -> None:
        self.args = args


class CloseStream:
    def __init__(self, msg: str = "Stream closed") -> None:
        self.msg = msg


def audio_to_float32(
    audio: NDArray[np.int16 | np.float32] | tuple[int, NDArray[np.int16 | np.float32]],
) -> NDArray[np.float32]:
    """
    Convert an audio tuple containing sample rate (int16) and numpy array data to float32.

    Parameters
    ----------
    audio : np.ndarray
        The audio data as a numpy array

    Returns
    -------
    np.ndarray
        The audio data as a numpy array with dtype float32

    Example
    -------
    >>> audio_data = np.array([0.1, -0.2, 0.3])  # Example audio samples
    >>> audio_float32 = audio_to_float32(audio_data)
    """
    if isinstance(audio, tuple):
        logging.warning(
            UserWarning(
                "Passing a (sr, audio) tuple to audio_to_float32() is deprecated "
                "and will be removed in a future release. Pass only the audio array."
            ),
            stacklevel=2,  # So that the warning points to the user's code
        )
        _sr, audio = audio

    if audio.dtype == np.int16:
        # Divide by 32768.0 so that the values are in the range [-1.0, 1.0).
        # 1.0 can actually never be reached because the int16 range is [-32768, 32767].
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float32:
        return audio  # type: ignore
    else:
        raise TypeError(f"Unsupported audio data type: {audio.dtype}")


async def wait_for_item(queue: asyncio.Queue, timeout: float = 0.1) -> tp.Any:
    """
    Wait for an item from an asyncio.Queue with a timeout.

    This function attempts to retrieve an item from the queue using asyncio.wait_for.
    If the timeout is reached, it returns None.

    This is useful to avoid blocking `emit` when the queue is empty.
    """

    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        return None
