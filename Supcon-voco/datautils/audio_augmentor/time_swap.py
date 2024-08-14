import torch
import torch.nn.functional as F
from .base import BaseAugmentor
from .utils import librosa_to_pydub
import numpy as np
import random
import librosa
import soundfile as sf
import os

import logging
logger = logging.getLogger(__name__)

class TimeSwapAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Time Swap augmentation
        Config:
        T: int, maximum time swap parameter
        num_swaps: int, number of time swaps to apply
        """
        super().__init__(config)
        self.T = config.get('T', 40)
        self.num_swaps = config.get('num_swaps', 1)  

    def apply_time_swapping(self, mel_spectrogram):
        """
        Apply time swapping to the mel spectrogram.
        """
        num_time_steps = mel_spectrogram.shape[1]

        for _ in range(self.num_swaps):
            t = random.randint(1, self.T)
            t0 = random.randint(0, num_time_steps - 2 * t)
            t1 = random.randint(t0 + t, num_time_steps - t)

            # Swap the segments
            segment1 = mel_spectrogram[:, t0:t0 + t].copy()
            segment2 = mel_spectrogram[:, t1:t1 + t].copy()
            mel_spectrogram[:, t0:t0 + t] = segment2
            mel_spectrogram[:, t1:t1 + t] = segment1

        return mel_spectrogram

    def transform(self):
        """
        Transform the audio by applying time swapping.
        """
        data = self.data.copy()
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr)

        swapped_mel_spectrogram = self.apply_time_swapping(mel_spectrogram)
        augmented_data = librosa.feature.inverse.mel_to_audio(swapped_mel_spectrogram, sr=self.sr)
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)