import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
import librosa
from datautils.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from datautils.audio_augmentor import BackgroundNoiseAugmentor, PitchAugmentor, ReverbAugmentor, SpeedAugmentor, VolumeAugmentor, TelephoneEncodingAugmentor, GaussianAugmentor, CopyPasteAugmentor, BaseAugmentor, TimeMaskingAugmentor, FrequencyMaskingAugmentor, MaskingAugmentor, TimeSwapAugmentor, FrequencySwapAugmentor, SwappingAugmentor, LinearFilterAugmentor, BandpassAugmentor
from datautils.audio_augmentor.utils import pydub_to_librosa, librosa_to_pydub
import soundfile as sf
import random

SUPPORTED_AUGMENTATION = [
    'background_noise_5_15', 'pitch_1', 'volume_10', 'reverb_1', 'speed_01', 'telephone_g722', 'gaussian_1', 'gaussian_2', 'gaussian_2_5', 'gaussian_3',
    'RawBoost12', 'copy_paste_80', 'copy_paste_r', 'time_masking', 'masking', 'time_swap', 
    'freq_swap', 'swapping', 'frequency_masking', 'linear_filter', 'mp32flac', 'ogg2flac', 'nonspeechtrim',
    'bandpass_0_4000', 'griffinlim_downsample', 'lowpass_hifigan_asvspoof5', 'lowpass_hifigan', 'librosa_downsample']



def audio_transform(filepath: str, aug_type: BaseAugmentor, config: dict, online: bool = False):
    """
    filepath: str, input audio file path
    aug_type: BaseAugmentor, augmentation type object
    config: dict, configuration dictionary
    online: bool, if True, return the augmented audio waveform, else save the augmented audio file
    """
    at = aug_type(config)
    at.load(filepath)
    at.transform()
    if online:
        audio = at.augmented_audio
        return pydub_to_librosa(audio)
    else: 
        at.save()
        
def background_noise_5_15(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    args.input_path = os.path.dirname(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'background_noise', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'background_noise')
    args.out_format = 'wav'
    config = {
        "aug_type": "background_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "noise_path": args.noise_path,
        "min_SNR_dB": 5,
        "max_SNR_dB": 15
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=BackgroundNoiseAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=BackgroundNoiseAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def pitch_1(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with pitch shift of -1 to 1
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'pitch', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'pitch')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "pitch",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_pitch_shift": -1,
        "max_pitch_shift": 1
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=PitchAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=PitchAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def volume_10(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with volume change of -10 to 10 dBFS
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'volume', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'volume')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "volume",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_volume_dBFS": -10,
        "max_volume_dBFS": 10
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=VolumeAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=VolumeAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
       
def reverb_1(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with reverb effect
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'reverb', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'reverb')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "reverb",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "rir_path": args.rir_path,
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=ReverbAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=ReverbAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def speed_01(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with speed change of 0.9 to 1.1
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'speed', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'speed')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "speed",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_speed_factor": 0.9,
        "max_speed_factor": 1.1
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=SpeedAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=SpeedAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def telephone_g722(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with telephone encoding g722 and bandpass filter
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'telephone', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'telephone')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "telephone",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "encoding": "g722",
        "bandpass": {
            "lowpass": "3400",
            "highpass": "400"
        }
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=TelephoneEncodingAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=TelephoneEncodingAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        
def bandpass_0_4000(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with bandpass filter of 0 to 4000 Hz
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'bandpass', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'bandpass')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "bandpass",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "lowpass": "4000",
        "highpass": "0"
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=BandpassAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=BandpassAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def gaussian_1(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with gaussian noise in the range of 0.001 to 0.015
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'gaussian_noise', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'gaussian_noise')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "guassian_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_amplitude": 0.001,
        "max_amplitude": 0.015
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=GaussianAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=GaussianAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def gaussian_2(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with gaussian noise in the range of 0.001 to 0.015
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'gaussian_noise', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'gaussian_noise')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "guassian_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_amplitude": 0.00001,
        "max_amplitude": 0.00015
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=GaussianAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=GaussianAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def gaussian_2_5(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with gaussian noise in the range of 0.001 to 0.015
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'gaussian_noise', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'gaussian_noise')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "guassian_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_amplitude": 0.000002,
        "max_amplitude": 0.00003
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=GaussianAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=GaussianAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def gaussian_3(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with gaussian noise in the range of 0.001 to 0.015
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'gaussian_noise', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'gaussian_noise')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "guassian_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_amplitude": 0.000001,
        "max_amplitude": 0.000015
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=GaussianAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=GaussianAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def copy_paste_r(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with copy paste of 80% of the audio, frame size is 800 samples
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'copy_paste', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'copy_paste')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "copy_paste",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "frame_size": 0,
        "shuffle_ratio": random.uniform(0.3, 1)
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=CopyPasteAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=CopyPasteAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def copy_paste_80(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with copy paste of 80% of the audio, frame size is 800 samples
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'copy_paste', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'copy_paste')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    config = {
        "aug_type": "copy_paste",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "frame_size": 800,
        "shuffle_ratio": 0.8
    }
    if (args.online_aug):
        waveform = audio_transform(filepath=audio_path, aug_type=CopyPasteAugmentor, config=config, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(filepath=audio_path, aug_type=CopyPasteAugmentor, config=config, online=False)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform      

        
def masking(x, args, sr=16000, audio_path=None):
    """
    Apply frequency and time masking to the audio file.
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'masking', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'masking')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "masking",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "F": 27,
        "num_frequency_masks": 2,
        "T": 100,
        "p": 1.0,
        "num_time_masks": 2
    }
    
    if args.online_aug:
        waveform = audio_transform(filepath=audio_path, aug_type=MaskingAugmentor, config=config, online=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            augmentor = MaskingAugmentor(config)
            augmentor.load(audio_path)
            augmentor.transform()
            augmentor.save(aug_audio_path)
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform


def time_swap(x, args, sr=16000, audio_path=None):
    """
    Apply time swapping to the audio file.
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'time_swap', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'time_swap')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "time_swap",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "T": 40,
        "num_swaps": 1   
    }
    
    if args.online_aug:
        waveform = audio_transform(filepath=audio_path, aug_type=TimeSwapAugmentor, config=config, online=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            augmentor = TimeSwapAugmentor(config)
            augmentor.load(audio_path)
            augmentor.transform()
            augmentor.save()
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        
def freq_swap(x, args, sr=16000, audio_path=None):
    """
    Apply frequency swapping to the audio file.
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'freq_swap', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'freq_swap')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "freq_swap",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "F": 7,
        "num_swaps": 1
    }
    
    if args.online_aug:
        waveform = audio_transform(filepath=audio_path, aug_type=FrequencySwapAugmentor, config=config, online=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            augmentor = FrequencySwapAugmentor(config)
            augmentor.load(audio_path)
            augmentor.transform()
            augmentor.save()
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        
def swapping(x, args, sr=16000, audio_path=None):
    """
    Apply time and frequency swapping to the audio file.
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'swapping', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'swapping')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "swapping",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "T": 40,
        "F": 7
    }
    
    if args.online_aug:
        waveform = audio_transform(filepath=audio_path, aug_type=SwappingAugmentor, config=config, online=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            augmentor = SwappingAugmentor(config)
            augmentor.load(audio_path)
            augmentor.transform()
            augmentor.save()
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def linear_filter(x, args, sr=16000, audio_path=None):
    """
    Apply linear filter augmentation to the audio file.
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'linear_filter', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'linear_filter')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "linear_filter",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "db_range": [-6, 6],
        "n_band": [3, 6],
        "min_bw": 6
    }
    
    if args.online_aug:
        waveform = audio_transform(filepath=audio_path, aug_type=LinearFilterAugmentor, config=config, online=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            augmentor = LinearFilterAugmentor(config)
            augmentor.load(audio_path)
            augmentor.transform()
            augmentor.save()
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
#--------------Offline only augmentation algorithms---------------------------##
def time_masking(x, args, sr=16000, audio_path=None):
    """
    Apply time masking to the audio file.
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'time_masking', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'time_masking')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "time_masking",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "T": 100,
        "p": 1.0,
        "num_masks": 2
    }
    if os.path.exists(aug_audio_path):
        waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        # augmentor = TimeMaskingAugmentor(config)
        # augmentor.load(audio_path)
        # augmentor.transform()
        # augmentor.save()
        # waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
        waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
        return waveform
    
def frequency_masking(x, args, sr=16000, audio_path=None):
    """
    Apply frequency masking to the audio file.
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'frequency_masking', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'frequency_masking')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    config = {
        "aug_type": "frequency_masking",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "F": 27,
        "num_masks": 2
    }
    # this aug do not support online augmentation
    if os.path.exists(aug_audio_path):
        waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        # augmentor = FrequencyMaskingAugmentor(config)
        # augmentor.load(audio_path)
        # augmentor.transform()
        # augmentor.save()
        # waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
        waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
        return waveform
    
def mp32flac(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with mp3 codec
    This codec is only available offline
    """
    # print('mp32flac: audio_path:', audio_path)
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_path = os.path.join(aug_dir, 'mp32flac', utt_id + '_from_mp3.flac')
    # check if the augmented file exists
    if (os.path.exists(aug_path)):
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
    else:
        # Alert ERROR and stop the process
        # print("Error: mp3 to flac conversion is only available offline\n Please convert to mp3 your data before running the script")
        # using ffmpeg to convert original audio to mp3
        os.system(f'ffmpeg -loglevel quiet -i {audio_path} {aug_path.replace(".flac", ".mp3")} -y')
        # using ffmpeg to convert mp3 to flac
        os.system(f'ffmpeg -loglevel quiet -i {aug_path.replace(".flac", ".mp3")} {aug_path} -y')
        # remove the mp3 file
        os.system(f'rm {aug_path.replace(".flac", ".mp3")}')
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
    
def ogg2flac(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with ogg codec
    This codec is only available offline
    """
    # print('ogg2flac: audio_path:', audio_path)
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_path = os.path.join(aug_dir, 'ogg2flac', utt_id + '_from_ogg.flac')
    # check if the augmented file exists
    if (os.path.exists(aug_path)):
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
    else:
        # Alert ERROR and stop the process
        # print("Error: ogg to flac conversion is only available offline\n Please convert to ogg your data before running the script")
        # using ffmpeg to convert original audio to ogg
        os.system(f'ffmpeg -loglevel quiet -i {audio_path} {aug_path.replace(".flac", ".ogg")} -y')
        # using ffmpeg to convert ogg to flac
        os.system(f'ffmpeg -loglevel quiet -i {aug_path.replace(".flac", ".ogg")} {aug_path} -y')
        # remove the ogg file
        os.system(f'rm {aug_path.replace(".flac", ".ogg")}')
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
        
def nonspeechtrim(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with non-speech trimming beginning and end
    This augmentation is only available offline
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_path = os.path.join(aug_dir, 'nonspeechtrim', utt_id + '.wav')
    
    if (os.path.exists(aug_path)):
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
    else:
        # Alert ERROR and stop the process
        # print("Error: nonspeech trimming is only available offline\n Please convert to oog your data before running the script")
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        waveform = librosa.effects.trim(y, top_db=10)[0]
        sf.write(aug_path, waveform, sr, subtype='PCM_16')
        return waveform

def griffinlim_downsample(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with neural codec
    This codec is only available offline
    """
    # print('ogg2flac: audio_path:', audio_path)
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_path = os.path.join(aug_dir, 'griffinlim_downsample', utt_id + '.wav')
    # check if the augmented file exists
    if (os.path.exists(aug_path)):
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
    else:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        # downsample to 8000
        y = librosa.resample(y, orig_sr=sr, target_sr=8000)
        y = librosa.resample(y, orig_sr=8000, target_sr=16000)
        # Griffin-Lim
        # print("D1")
        S = np.abs(librosa.stft(y))
        # print("D2")
        y_inv = librosa.griffinlim(S)
        sf.write(aug_path, y_inv, sr, subtype='PCM_16')
        return y_inv
    
def librosa_downsample(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with neural codec
    This codec is only available offline
    """
    # print('ogg2flac: audio_path:', audio_path)
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_path = os.path.join(aug_dir, 'librosa_downsample', utt_id + '.wav')
    # check if the augmented file exists
    if (os.path.exists(aug_path)):
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
    else:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        # downsample to 8000
        y = librosa.resample(y, orig_sr=sr, target_sr=8000)
        y = librosa.resample(y, orig_sr=8000, target_sr=16000)
        sf.write(aug_path, y, sr, subtype='PCM_16')
        return y

def lowpass_hifigan_asvspoof5(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with neural codec
    This codec is only available offline
    """
    # print('ogg2flac: audio_path:', audio_path)
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_path = os.path.join(aug_dir, 'lowpass_hifigan_asvspoof5', utt_id + '.wav')
    # check if the augmented file exists
    if (os.path.exists(aug_path)):
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
    else:
        # just load the original audio
        return x

def lowpass_hifigan(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with neural codec
    This codec is only available offline
    """
    # print('ogg2flac: audio_path:', audio_path)
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_path = os.path.join(aug_dir, 'lowpass_hifigan', utt_id + '.wav')
    # check if the augmented file exists
    if (os.path.exists(aug_path)):
        waveform,_ = librosa.load(aug_path, sr=sr, mono=True)
        return waveform
    else:
        # just load the original audio
        return x

#--------------RawBoost data augmentation algorithms---------------------------##

def RawBoost12(x, args, sr = 16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'RawBoost12', utt_id + '.wav')
    if args.online_aug:
        return process_Rawboost_feature(x, sr,args, algo=5)
    else:
        # check if the augmented file exists
        if (os.path.exists(aug_audio_path)):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            waveform= process_Rawboost_feature(x, sr,args, algo=5)
            # save the augmented file,waveform in np array
            sf.write(aug_audio_path, waveform, sr, subtype='PCM_16')
            return waveform

def RawBoostFull(x, args, sr = 16000, audio_path = None):
    '''
        Apply all 3 algo. together in series (1+2+3)
    '''
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    aug_audio_path = os.path.join(aug_dir, 'RawBoostFull', utt_id + '.wav')
    if args.online_aug:
        return process_Rawboost_feature(x, sr,args, algo=4)
    else:
        # check if the augmented file exists
        if (os.path.exists(aug_audio_path)):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            waveform= process_Rawboost_feature(x, sr,args, algo=4)
            # save the augmented file,waveform in np array
            sf.write(aug_audio_path, waveform, sr, subtype='PCM_16')
            return waveform
            
def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature

