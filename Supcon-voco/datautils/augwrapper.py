import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
import librosa
from datautils.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from datautils.audio_augmentor import BackgroundNoiseAugmentor, PitchAugmentor, ReverbAugmentor, SpeedAugmentor, VolumeAugmentor, TelephoneEncodingAugmentor, GaussianAugmentor
from datautils.audio_augmentor.utils import pydub_to_librosa, librosa_to_pydub
import soundfile as sf

def background_noise(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "background_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "noise_path": args.noise_path,
        "min_SNR_dB": 5,
        "max_SNR_dB": 15
    }
    bga = BackgroundNoiseAugmentor(config)
    
    bga.load(in_file)
    bga.transform()
    if online:
        audio = bga.augmented_audio
        return pydub_to_librosa(audio)
    else: 
        bga.save()
    

def pitch(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "pitch",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_pitch_shift": -1,
        "max_pitch_shift": 1
    }
    pa = PitchAugmentor(config)
    pa.load(in_file)
    pa.transform()
    if online:
        audio = pa.augmented_audio
        return pydub_to_librosa(audio)
    else:
        pa.save()
    
    
def reverb(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "reverb",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "rir_path": args.rir_path,
    }
    ra = ReverbAugmentor(config)
    ra.load(in_file)
    ra.transform()
    if online:
        audio = ra.augmented_audio
        return pydub_to_librosa(audio)
    else:
        ra.save()

def speed(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "speed",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_speed_factor": 0.9,
        "max_speed_factor": 1.1
    }
    sa = SpeedAugmentor(config)
    sa.load(in_file)
    sa.transform()
    if online:
        audio = sa.augmented_audio
        return pydub_to_librosa(audio)
    else:
        sa.save()

    
def volume(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "volume",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_volume_dBFS": -10,
        "max_volume_dBFS": 10
    }
    va = VolumeAugmentor(config)
    va.load(in_file)
    va.transform()
    if online:
        audio = va.augmented_audio
        return pydub_to_librosa(audio)
    else:
        va.save()

def telephone(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "telephone",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "encoding": "ALAW",
        "bandpass": {
            "lowpass": "3400",
            "highpass": "400"
        }
    }
    ta = TelephoneEncodingAugmentor(config)
    ta.load(in_file)
    ta.transform()
    if online:
        audio = ta.augmented_audio
        return pydub_to_librosa(audio)
    else:
        ta.save()
        
def gaussian_noise(args, filename, online = False):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "guassian_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_amplitude": 0.001,
        "max_amplitude": 0.015
    }
    gn = GaussianAugmentor(config)
    gn.load(in_file)
    gn.transform()
    if online:
        audio = gn.augmented_audio
        return pydub_to_librosa(audio)
    else:
        gn.save()

def background_noise_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)

    aug_audio_path = os.path.join(aug_dir, 'background_noise', utt_id)
    args.output_path = os.path.join(aug_dir, 'background_noise')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = background_noise(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            background_noise(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def pitch_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'pitch', utt_id)
    args.output_path = os.path.join(aug_dir, 'pitch')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = pitch(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            waveform = pitch(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def volume_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'volume', utt_id)
    args.output_path = os.path.join(aug_dir, 'volume')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = volume(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            waveform = volume(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
    
def reverb_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'reverb', utt_id)
    args.output_path = os.path.join(aug_dir, 'reverb')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = reverb(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            reverb(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def speed_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'speed', utt_id)
    args.output_path = os.path.join(aug_dir, 'speed')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = speed(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            speed(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

def telephone_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'telephone', utt_id)
    args.output_path = os.path.join(aug_dir, 'telephone')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = telephone(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            telephone(args, utt_id)
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        
def gaussian_noise_wrapper(x, args, sr=16000, audio_path = None):
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'gaussian_noise', utt_id)
    args.output_path = os.path.join(aug_dir, 'gaussian_noise')
    args.out_format = 'wav'
    args.input_path = os.path.dirname(audio_path)
    
    if (args.online_aug):
        waveform = gaussian_noise(args, utt_id, online=True)
        # waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform,_ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            telephone(args, utt_id)
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
