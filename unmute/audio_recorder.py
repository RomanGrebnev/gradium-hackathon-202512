import logging
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, recordings_dir: Path, sample_rate: int = 24000):
        self.recordings_dir = recordings_dir
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_id = uuid.uuid4().hex[:4]
        self.filename = self.recordings_dir / f"audio_log_{timestamp}_{unique_id}.wav"
        
        self.wav_file = wave.open(str(self.filename), 'wb')
        self.wav_file.setnchannels(1) # Mono
        self.wav_file.setsampwidth(2) # 16-bit
        self.wav_file.setframerate(self.sample_rate)
        
        logger.info(f"Audio recording started: {self.filename}")

    def add_audio(self, audio_chunk: np.ndarray):
        """
        Appends float32 or int16 numpy array audio chunk to the file.
        Assumes the chunk matches the sample_rate.
        """
        if self.wav_file is None:
            return

        try:
            # Convert to int16 if float
            if audio_chunk.dtype == np.float32:
                # Clip and scale
                audio_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
            elif audio_chunk.dtype == np.int16:
                audio_int16 = audio_chunk
            else:
                # Try casting
                audio_int16 = audio_chunk.astype(np.int16)
            
            self.wav_file.writeframes(audio_int16.tobytes())
            
        except Exception as e:
            logger.error(f"Error writing audio chunk: {e}")

    def shutdown(self):
        if self.wav_file:
            try:
                self.wav_file.close()
                logger.info(f"Audio recording saved: {self.filename}")
                self.wav_file = None
            except Exception as e:
                logger.error(f"Error closing audio file: {e}")
