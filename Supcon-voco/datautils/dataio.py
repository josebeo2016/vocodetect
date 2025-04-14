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

def process_audio_samples(audio_list, trim_length, padding_type='zero'):
    processed_audios = []

    for audio in audio_list:
        processed_audios.append(pad(audio, padding_type=padding_type, max_len=trim_length))
    return processed_audios

def multiview_process_audio(audio_list, audio_paths, args, trim_length=64000, sr=16000, augmentation_func=None, padding_type='zero'):
    if audio_paths is None or len(audio_paths) != len(audio_list):
        raise ValueError("audio_paths must be provided and match the length of audio_list.")

    # Ensure all input audios are the same length
    original_audios = process_audio_samples(audio_list, trim_length, padding_type=padding_type)
    
    augmented_audios = []
    
    for i, (audio, path) in enumerate(zip(original_audios, audio_paths)):
        # Augment the audio using the specified function
        aug_audio = augmentation_func(audio, args, sr=sr, audio_path=path)
        
        # Check if augmented audio is the same length as the original
        if aug_audio.shape[0] != trim_length:
            # Trim or zero pad to make it equal
            aug_audio = process_audio_samples([aug_audio], trim_length, padding_type=padding_type)[0]
        
        augmented_audios.append(aug_audio)
    
    # Create a multiview tensor
    multiview_tensor = np.stack([original_audios, augmented_audios], axis=1)
    
    return multiview_tensor # [num_samples, num_views, trim_length]

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

