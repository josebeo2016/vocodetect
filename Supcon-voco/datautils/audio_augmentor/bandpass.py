from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
import scipy.signal as ss
import librosa
import random
import numpy as np
import torchaudio.functional as F
from torchaudio.io import AudioEffector
import logging
import torch
from torchaudio.sox_effects import apply_effects_tensor as sox_fx

logger = logging.getLogger(__name__)

def apply_codec(waveform, sample_rate, format, encoder=None):
    encoder = AudioEffector(format=format, encoder=encoder)
    return encoder.apply(waveform, sample_rate)

def apply_effect(waveform, sample_rate, effect):
    effector = AudioEffector(effect=effect)
    return effector.apply(waveform, sample_rate)
class BandpassAugmentor(BaseAugmentor):
    """
    About
    -----

    This class makes audio sound like it's over telephone, by apply one or two things:
    * codec: choose between ALAW, ULAW, g722
    * bandpass: Optional, can be None. This limits the frequency within common telephone range (300, 3400) be default.
                Note: lowpass value should be higher than that of highpass.

    Example
    -------

    CONFIG = {
        "aug_type": "telephone",
        "output_path": os.path.join(BASE_DIR,"data/augmented"),
        "out_format": "wav",
        "encoding": "ALAW",
        "bandpass": {
            "lowpass": "4000",
            "highpass": "0"
        }
    }
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.bandpass = {
            "lowpass": config.get("lowpass", 4000),
            "highpass": config.get("highpass", 0)
        }

        self.effects = ",".join(
            [
                "lowpass=frequency={}:poles=1".format(self.bandpass['lowpass']),
                "highpass=frequency={}:poles=1".format(self.bandpass['highpass']),
            ]
        )
        
    def transform(self):
        """
        """
        torch_audio = torch.tensor(self.data).reshape(1, -1)
        aug_audio = apply_effect(torch_audio.T, self.sr, self.effects)
        # convert to numpy array
        aug_audio = aug_audio.numpy().flatten()
        self.augmented_audio = librosa_to_pydub(aug_audio)