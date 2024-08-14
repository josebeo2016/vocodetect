from .base import BaseAugmentor
from .utils import librosa_to_pydub
import numpy as np
import random
import librosa
import soundfile as sf
import os

import logging
logger = logging.getLogger(__name__)

class FrequencyMaskingAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Frequency Masking augmentation
        Config:
        F: int, maximum frequency mask parameter
        num_masks: int, number of frequency masks to apply
        """
        super().__init__(config)
        self.F = config.get('F', 27)
        self.num_masks = config.get('num_masks', 2)

    def apply_frequency_masking(self, mel_spectrogram):
        """
        Apply frequency masking to the mel spectrogram.
        """
        num_mel_channels = mel_spectrogram.shape[0]
        
        for _ in range(self.num_masks):
            f = random.randint(0, self.F)
            f0 = random.randint(0, num_mel_channels - f)
            mel_spectrogram[f0:f0 + f, :] = 0
        
        return mel_spectrogram

    def transform(self):
        """
        Transform the audio by applying frequency masking.
        """
        data = self.data.copy()
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr)
        masked_mel_spectrogram = self.apply_frequency_masking(mel_spectrogram)
        augmented_data = librosa.feature.inverse.mel_to_audio(masked_mel_spectrogram, sr=self.sr)
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)