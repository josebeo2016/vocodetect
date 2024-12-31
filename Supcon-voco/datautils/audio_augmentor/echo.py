from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
import scipy.signal as ss
import librosa
import random
import numpy as np
import logging
import torch


logger = logging.getLogger(__name__)

def add_echo(audio: np.ndarray, sample_rate: int, delay=0.5, decay=0.3) -> np.ndarray:
    # Calculate the number of samples for the delay
    delay_samples = int(sample_rate * delay)
    
    # Create an empty array for the echoed audio
    echoed_audio = np.zeros(len(audio) + delay_samples)
    
    # Add the original audio
    echoed_audio[:len(audio)] += audio
    
    # Add the echo
    echoed_audio[delay_samples:] += decay * audio
    
    return echoed_audio

class EchoAugmentor(BaseAugmentor):
    """
    About
    -----

    This class makes audio sound has echo effect.
    some dramatic feeling can be added to the audio.
    
    Example
    -------

    CONFIG = {
        "aug_type": "echo",
        "output_path": os.path.join(BASE_DIR,"data/augmented"),
        "out_format": "wav",
        "min_delay": 100,
        "max_delay": 1000,
        "min_decay": 0.3,
        "max_decay": 0.9,
    }
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_delay = config.get("min_delay", 100)
        self.max_delay = config.get("max_delay", 1000)
        self.min_decay = config.get("min_decay", 0.3)
        self.max_decay = config.get("max_decay", 0.9)
        
    def transform(self):
        """
        """
        delay = random.randint(self.min_delay, self.max_delay)
        decay = random.uniform(self.min_decay, self.max_decay)
        aug_audio = add_echo(self.data, self.sr, delay, decay)

        self.augmented_audio = librosa_to_pydub(aug_audio)