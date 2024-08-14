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

class FrequencySwapAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Frequency Swap augmentation
        Config:
        F: int, maximum frequency swap parameter
        num_swaps: int, number of frequency swaps to apply
        """
        super().__init__(config)
        self.F = config.get('F', 7)
        self.num_swaps = config.get('num_swaps', 1)

    def apply_frequency_swapping(self, mel_spectrogram):
        """
        Apply frequency swapping to the mel spectrogram.
        """
        num_mel_channels = mel_spectrogram.shape[0]

        for _ in range(self.num_swaps):
            f = random.randint(0, self.F)
            f0 = random.randint(0, num_mel_channels - 2 * f)
            f1 = random.randint(f0 + f, num_mel_channels - f)

            mel_spectrogram[f0:f0 + f], mel_spectrogram[f1:f1 + f] = \
            mel_spectrogram[f1:f1 + f].clone(), mel_spectrogram[f0:f0 + f].clone()

        return mel_spectrogram

    def transform(self):
        """
        Transform the audio by applying frequency swapping.
        """
        data = self.data.copy()
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr)

        swapped_mel_spectrogram = self.apply_frequency_swapping(torch.tensor(mel_spectrogram))
        swapped_mel_spectrogram = swapped_mel_spectrogram.numpy()
        
        augmented_data = librosa.feature.inverse.mel_to_audio(swapped_mel_spectrogram, sr=self.sr)
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)