import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
import librosa
from datautils.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from datautils.audio_augmentor import BackgroundNoiseAugmentor, PitchAugmentor, ReverbAugmentor, SpeedAugmentor, VolumeAugmentor, TelephoneEncodingAugmentor, GaussianAugmentor, CopyPasteAugmentor, BaseAugmentor
from datautils.audio_augmentor.utils import pydub_to_librosa, librosa_to_pydub
import soundfile as sf
import random

SUPPORTED_AUGMENTATION = ['background_noise_5_15', 'pitch_1', 'volume_10', 'reverb_1', 'speed_01', 'telephone_g722', 'gaussian_1', 'RawBoost12', 'copy_paste_80', 'copy_paste_r']

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
    utt_id = os.path.basename(audio_path)
    args.input_path = os.path.dirname(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'background_noise', utt_id)
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
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'pitch', utt_id)
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
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'volume', utt_id)
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
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'reverb', utt_id)
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
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'speed', utt_id)
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
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'telephone', utt_id)
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
        
def gaussian_1(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with gaussian noise in the range of 0.001 to 0.015
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'gaussian_noise', utt_id)
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

def copy_paste_80(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with copy paste of 80% of the audio, frame size is 800 samples
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'copy_paste', utt_id)
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
        
def copy_paste_r(x, args, sr=16000, audio_path = None):
    """
    Augment the audio with copy paste of 80% of the audio, frame size is 800 samples
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'copy_paste', utt_id)
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
#--------------RawBoost data augmentation algorithms---------------------------##

def RawBoost12(x, args, sr = 16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'RawBoost12', utt_id)
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

