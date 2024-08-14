import torch
import numpy as np
import random
import librosa
from .base import BaseAugmentor
from .utils import librosa_to_pydub

import logging
logger = logging.getLogger(__name__)

class LinearFilterAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Linear Filter augmentation
        Config:
        db_range: list, range of dB values for filter weights
        n_band: list, range of number of frequency bands
        min_bw: int, minimum bandwidth
        """
        super().__init__(config)
        self.db_range = config.get('db_range', [-6, 6])
        self.n_band = config.get('n_band', [3, 6])
        self.min_bw = config.get('min_bw', 6)

    def apply_linear_filter(self, mel_spectrogram):
        """
        Apply linear filter to the mel spectrogram.
        """
        mel_spectrogram = torch.tensor(mel_spectrogram)  # Convert numpy array to torch tensor
        n_freq_bin, time_steps = mel_spectrogram.shape
        n_freq_band = random.randint(self.n_band[0], self.n_band[1])
        
        if n_freq_band > 1:
            while n_freq_bin - n_freq_band * self.min_bw + 1 < 0:
                self.min_bw -= 1
            band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * self.min_bw + 1,
                                                        (n_freq_band - 1,)))[0] + \
                               torch.arange(1, n_freq_band) * self.min_bw
            band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

            band_factors = torch.rand(n_freq_band + 1) * (self.db_range[1] - self.db_range[0]) + self.db_range[0]
            freq_filt = torch.ones((n_freq_bin, 1))
            for i in range(n_freq_band):
                freq_filt[band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                    torch.linspace(band_factors[i], band_factors[i+1],
                                   band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
            return mel_spectrogram * freq_filt
        else:
            return mel_spectrogram

    def transform(self):
        """
        Transform the audio by applying linear filter.
        """
        data = self.data.copy()
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr)
        filtered_mel_spectrogram = self.apply_linear_filter(mel_spectrogram)
        filtered_audio = librosa.feature.inverse.mel_to_audio(filtered_mel_spectrogram.numpy(), sr=self.sr)  # Convert back to numpy array
        self.augmented_audio = librosa_to_pydub(filtered_audio, sr=self.sr)