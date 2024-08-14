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

class SwappingAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Swapping augmentation
        Config:
        T: int, maximum time swap parameter
        F: int, maximum frequency swap parameter
        """
        super().__init__(config)
        self.T = config.get('T', 40)
        self.F = config.get('F', 7)

    def apply_time_swapping(self, mel_spectrogram):
        """
        Apply time swapping to the mel spectrogram.
        """
        num_time_steps = mel_spectrogram.shape[1]

        t = random.randint(0, self.T)
        t0 = random.randint(0, num_time_steps - 2 * t)
        t1 = random.randint(t0 + t, num_time_steps - t)

        mel_spectrogram[:, t0:t0 + t], mel_spectrogram[:, t1:t1 + t] = \
            mel_spectrogram[:, t1:t1 + t].copy(), mel_spectrogram[:, t0:t0 + t].copy()

        return mel_spectrogram

    def apply_frequency_swapping(self, mel_spectrogram):
        """
        Apply frequency swapping to the mel spectrogram.
        """
        num_mel_channels = mel_spectrogram.shape[0]

        f = random.randint(0, self.F)
        f0 = random.randint(0, num_mel_channels - 2 * f)
        f1 = random.randint(f0 + f, num_mel_channels - f)

        mel_spectrogram[f0:f0 + f, :], mel_spectrogram[f1:f1 + f, :] = \
            mel_spectrogram[f1:f1 + f, :].copy(), mel_spectrogram[f0:f0 + f, :].copy()

        return mel_spectrogram

    def transform(self):
        """
        Transform the audio by applying time and frequency swapping.
        """
        data = self.data.copy()
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr)
        
        mel_spectrogram = self.apply_time_swapping(mel_spectrogram)
        swapped_mel_spectrogram = self.apply_frequency_swapping(mel_spectrogram)

        augmented_data = librosa.feature.inverse.mel_to_audio(swapped_mel_spectrogram, sr=self.sr)
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)