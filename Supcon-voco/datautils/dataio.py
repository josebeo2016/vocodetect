import os
import numpy as np
import librosa
import soundfile as sf
import torch
#########################
# Set up logging
#########################
import logging
from logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def load_audio(file_path: str, sr: int=16000) -> np.ndarray:
    '''
    Load audio file
    file_path: path to the audio file
    sr: sampling rate, default 16000
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def save_audio(file_path: str, audio: np.ndarray, sr: int=16000):
    '''
    Save audio file
    file_path: path to save the audio file
    audio: audio signal
    sr: sampling rate, default 16000
    '''
    sf.write(file_path, audio, sr, subtype='PCM_16')
    
def npwav2torch(waveform: np.ndarray) -> torch.Tensor:
    '''
    Convert numpy array to torch tensor
    waveform: audio signal
    '''
    return torch.from_numpy(waveform).float()

def pad(x:np.ndarray, padding_type:str='zero', max_len=64000, random_start=False) -> np.ndarray:
        '''
        pad audio signal to max_len
        x: audio signal
        padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
            zero: pad with zeros
            repeat: repeat the signal until it reaches max_len
        max_len: max length of the audio, default 64000
        random_start: if True, randomly choose the start point of the audio
        '''
        x_len = x.shape[0]
        padded_x = None
        if max_len == 0:
            # no padding
            padded_x = x
        elif max_len > 0:
            if x_len >= max_len:
                if random_start:
                    start = np.random.randint(0, x_len - max_len+1)
                    padded_x = x[start:start + max_len]
                    # logger.debug("padded_x1: {}".format(padded_x.shape))
                else:
                    padded_x = x[:max_len]
                    # logger.debug("padded_x2: {}".format(padded_x.shape))
            else:
                if random_start:
                    # keep at least half of the signal
                    start = np.random.randint(0, int((x_len+1)/2))
                    x_new = x[start:]
                else:
                    x_new = x
                
                if padding_type == "repeat":
                    num_repeats = int(max_len / len(x_new)) + 1
                    padded_x = np.tile(x_new, (1, num_repeats))[:, :max_len][0]

                elif padding_type == "zero":
                    padded_x = np.zeros(max_len)
                    padded_x[:len(x_new)] = x_new

        else:
            raise ValueError("max_len must be >= 0")
        # logger.debug("padded_x: {}".format(padded_x.shape))
        return padded_x

