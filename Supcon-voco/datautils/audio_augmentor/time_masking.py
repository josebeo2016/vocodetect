from .base import BaseAugmentor
from .utils import librosa_to_pydub
import numpy as np
import random
import librosa
import soundfile as sf
import os

import logging
logger = logging.getLogger(__name__)

class TimeMaskingAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Time Masking augmentation
        Config:
        T: int, maximum time mask parameter (number of frames to mask)
        p: float, upper bound parameter for time mask as a fraction of total time steps
        num_masks: int, number of time masks to apply
        """
        super().__init__(config)
        self.T = config.get('T', 100)
        self.p = config.get('p', 1.0)
        self.num_masks = config.get('num_masks', 2)

    def apply_time_masking(self, mel_spectrogram):
        """
        Apply time masking to the mel spectrogram.
        """
        num_time_steps = mel_spectrogram.shape[1]
        max_mask_length = int(self.p * num_time_steps)

        for _ in range(self.num_masks):
            if num_time_steps > 1:
                t = random.randint(1, min(self.T, max_mask_length))
                t0 = random.randint(0, num_time_steps - t)
                mel_spectrogram[:, t0:t0 + t] = 0

        return mel_spectrogram

    def transform(self):
        """
        Transform the audio by applying time masking.
        """
        data = self.data.copy()
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr)

        # Apply time masking
        masked_mel_spectrogram = self.apply_time_masking(mel_spectrogram)
        masked_audio = librosa.feature.inverse.mel_to_audio(masked_mel_spectrogram, sr=self.sr)
        self.augmented_audio = librosa_to_pydub(masked_audio, sr=self.sr)
