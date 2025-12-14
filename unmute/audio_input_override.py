from pathlib import Path

import numpy as np
import sphn

from unmute.gradium_constants import STT_SAMPLE_RATE


class AudioInputOverride:
    def __init__(self, file: Path):
        data, _ = sphn.read(file, sample_rate=STT_SAMPLE_RATE)
        assert data.ndim == 2

        if data.dtype != np.int16:
            data = (data * np.iinfo(np.int16).max).astype(np.int16)

        self.data = data
        self.position = 0

    def override(self, original_data: np.ndarray) -> np.ndarray:
        if self.position + original_data.shape[1] > self.data.shape[1]:
            return original_data

        data = self.data[
            :, self.position : self.position + original_data.shape[1]
        ].copy()
        self.position += original_data.shape[1]

        assert data.shape == original_data.shape, (
            f"{data.shape} != {original_data.shape}"
        )
        assert data.dtype == original_data.dtype

        return data
