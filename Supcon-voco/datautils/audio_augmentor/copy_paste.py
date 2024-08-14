from .base import BaseAugmentor
from .utils import librosa_to_pydub
import random
import soundfile as sf
import numpy as np
import logging

logger = logging.getLogger(__name__)
class CopyPasteAugmentor(BaseAugmentor):
    """
        Copy and Patse augmentation
        Config:
            shuffle_ratio (float): The ratio of frames to shuffle (between 0 and 1).
            frame_size (int): The size of each frame. If zero, the frame size is randomly selected from [800:3200] samples.
            
    """
    def __init__(self, config: dict):
        """
        This method initialize the `GaussianAugmentor` object.
        
        :param config: dict, configuration dictionary
        """
        
        super().__init__(config)
        self.shuffle_ratio = config['shuffle_ratio']
        self.frame_size = config['frame_size']
        assert self.shuffle_ratio > 0.0 and self.shuffle_ratio < 1.0
        assert self.frame_size >= 0
        # Determine frame size
        if self.frame_size == 0:
            self.frame_size = np.random.randint(800, 3201)  # Randomly select frame size from [800:3200]

    def transform(self):
        """
        Transform the audio by adding gausian noise.
        """
        
        # Check if the frame_size is greater than the audio length
        if self.frame_size > len(self.data):
            logger.warning(f"Frame size {self.frame_size} is greater than audio length {len(self.data)}. Skipping augmentation.")
            # no transformation, just return the original audio
            self.augmented_audio = librosa_to_pydub(self.data, sr=self.sr)
            return
        
        # Split audio into frames
        num_frames = len(self.data) // self.frame_size
        frames = [self.data[i*self.frame_size:(i+1)*self.frame_size] for i in range(num_frames)]
        
        # Handle the last chunk of audio if it's not a full frame
        if len(self.data) % self.frame_size != 0:
            frames.append(self.data[num_frames*self.frame_size:])
        
        # Determine the number of frames to shuffle
        num_frames_to_shuffle = int(len(frames) * self.shuffle_ratio)
        
        # Randomly shuffle a ratio of frames
        if num_frames_to_shuffle > 0:
            shuffle_indices = np.random.choice(len(frames), num_frames_to_shuffle, replace=False)
            shuffled_frames = np.array(frames, dtype=object)[shuffle_indices]
            np.random.shuffle(shuffled_frames)
            for i, idx in enumerate(shuffle_indices):
                frames[idx] = shuffled_frames[i]
        
        # Concatenate frames back together
        transformed_audio = np.concatenate(frames)
        # transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(transformed_audio, sr=self.sr)
