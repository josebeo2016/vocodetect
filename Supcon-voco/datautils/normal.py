import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import librosa
from core_scripts.data_io import wav_augmentation as nii_wav_aug
from core_scripts.data_io import wav_tools as nii_wav_tools
from datautils.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import logging

logging.basicConfig(filename='errors.log', level=logging.DEBUG)
def genList(dir_meta, is_train=False, is_eval=False, is_dev=False):
    # bonafide: 1, spoof: 0
    d_meta = {}
    file_list=[]
    dir_meta = os.path.dirname(dir_meta)
    protocol = os.path.join(dir_meta, "protocol.txt")
    with open(protocol, 'r') as f:
        l_meta = f.readlines()
    if (is_train):
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'train':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0

        return d_meta, file_list
    if (is_dev):
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'dev':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    if (is_eval):
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'eval':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0
        # return d_meta, file_list
        return d_meta, file_list
    
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

    # init padded_x
    padded_x = None
    if max_len == 0:
        # no padding
        padded_x = x
    elif max_len > 0:
        if x_len >= max_len:
            if random_start:
                start = np.random.randint(0, x_len - max_len+1)
                padded_x = x[start:start + max_len]
            else:
                padded_x = x[:max_len]
        else:
            if padding_type == "repeat":
                num_repeats = int(max_len / x_len) + 1
                padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]

            elif padding_type == "zero":
                padded_x = np.zeros(max_len)
                padded_x[:x_len] = x
    else:
        raise ValueError("max_len must be >= 0")
    return padded_x

class Dataset_for(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[], 
                 augmentation_methods=[], num_additional_real=2, num_additional_spoof=2, 
                 trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None, 
                 aug_dir=None, online_aug=False, repeat_pad=True, is_train=True, **kwargs):
        """
        Args:
            list_IDs (string): Path to the .lst file with real audio filenames.
            vocoders (list): list of vocoder names.
            augmentation_methods (list): List of augmentation methods to apply.
        """
        self.args = args
        self.algo = algo
        self.labels = labels
        self.args.noise_path = noise_path
        self.args.rir_path = rir_path
        self.args.aug_dir = aug_dir
        self.args.online_aug = online_aug
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.bonafide_dir = os.path.join(base_dir, 'bonafide')
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        
        self.trim_length = trim_length
        self.sample_rate = wav_samp_rate
        self.repeat_pad = repeat_pad

        self.vocoders = vocoders
        print("vocoders:", vocoders)
        self.num_additional_real = num_additional_real
        self.augmentation_methods = augmentation_methods
        
        if len(augmentation_methods) < 1:
            # using default augmentation method RawBoostWrapper12
            self.augmentation_methods = ["RawBoost12"]
    
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X, fs = librosa.load(filepath, sr=16000)
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_pad= pad(Y,padding_type="repeat" if self.repeat_pad else "zero", max_len=self.trim_length, random_start=False)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        return idx, x_inp, target

class Dataset_for_dev(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[], 
                 augmentation_methods=[], num_additional_real=2, num_additional_spoof=2, 
                 trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None, 
                 aug_dir=None, online_aug=False, repeat_pad=True, is_train=True, **kwargs):
        self.list_IDs = list_IDs
        self.base_dir = os.path.join(base_dir)
        self.cut=trim_length 
        self.labels = labels
        if repeat_pad:
            self.padding_type = "repeat"
        else:
            self.padding_type = "zero"
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        X_pad = pad(X,self.padding_type,self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return index, x_inp, target

class Dataset_for_eval(Dataset):
    def __init__(self, list_IDs, base_dir, padding_type='zero', max_len=64000):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.padding_type = padding_type
        self.trim_length=max_len # take ~4 sec audio (64600 samples)
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
            
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X, _ = librosa.load(filepath, sr=16000)
        X_pad= pad(X,padding_type=self.padding_type,max_len=self.trim_length, random_start=False)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


# ------------------audio augmentor wrappers---------------------------##
from .audio_augmentor import BackgroundNoiseAugmentor, PitchAugmentor, ReverbAugmentor, SpeedAugmentor, VolumeAugmentor
from .audio_augmentor.utils import pydub_to_librosa, librosa_to_pydub

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


#--------------RawBoost data augmentation algorithms---------------------------##
import soundfile as sf
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
