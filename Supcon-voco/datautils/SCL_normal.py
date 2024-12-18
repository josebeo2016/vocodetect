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
import random
# augwrapper
from datautils.augwrapper import background_noise_wrapper, pitch_wrapper, reverb_wrapper, speed_wrapper, volume_wrapper, telephone_wrapper, gaussian_noise_wrapper, RawBoost12

logging.basicConfig(filename='errors.log', level=logging.DEBUG)
def genList(dir_meta, is_train=False, is_eval=False, is_dev=False):
    # bonafide: 1, spoof: 0
    d_meta = {}
    file_list=[]
    # get dir of metafile only
    dir_meta = os.path.dirname(dir_meta)
    if is_train:
        metafile = os.path.join(dir_meta, 'scp/train_bonafide.lst')
    elif is_dev:
        metafile = os.path.join(dir_meta, 'scp/dev_bonafide.lst')
    elif is_eval:
        metafile = os.path.join(dir_meta, 'scp/test.lst')
        
    with open(metafile, 'r') as f:
        l_meta = f.readlines()
    
    if (is_train):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])
        return [],file_list
    
    if (is_dev):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])
        return [],file_list
    
    elif(is_eval):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])

        return [], file_list
    
def pad(x, padding_type, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    if padding_type == "repeat":
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    elif padding_type == "zero":
        padded_x = np.zeros(max_len)
        padded_x[:x_len] = x
    return padded_x

class Dataset_for(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[], 
                 augmentation_methods=[], num_additional_real=2, num_additional_spoof=2, 
                 trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None, 
                 aug_dir=None, online_aug=False, repeat_pad=True):
        """
        Args:
            list_IDs (string): Path to the .lst file with real audio filenames.
            vocoders (list): list of vocoder names.
            augmentation_methods (list): List of augmentation methods to apply.
        """
        self.args = args
        self.is_train = args.is_train
        self.args.noise_path = noise_path
        self.args.rir_path = rir_path
        self.args.aug_dir = aug_dir
        self.args.online_aug = online_aug
        self.list_IDs = list_IDs
        self.bonafide_dir = os.path.join(base_dir, 'bonafide')
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')

        # list available spoof samples (only .wav files)
        if self.is_train:
            self.spoof_dir = os.path.join(base_dir, 'spoof_train')
        else:
            self.spoof_dir = os.path.join(base_dir, 'spoof_dev')
        self.spoof_list = [f for f in os.listdir(self.spoof_dir) if os.path.isfile(os.path.join(self.spoof_dir, f)) and (f.endswith('.wav') or f.endswith('.flac'))]
        self.repeat_pad = repeat_pad
        self.trim_length = trim_length
        self.sample_rate = wav_samp_rate

        self.vocoders = vocoders
        print("vocoders:", vocoders)
        self.num_additional_spoof = num_additional_spoof
        self.num_additional_real = num_additional_real
        self.augmentation_methods = augmentation_methods
        
        if len(augmentation_methods) < 1:
            # using default augmentation method RawBoostWrapper12
            self.augmentation_methods = ["RawBoost12"]

    def load_audio(self, file_path):
        waveform,_ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        # _, waveform = nii_wav_tools.waveReadAsFloat(file_path)
        return waveform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # Anchor real audio sample
        real_audio_file = os.path.join(self.bonafide_dir, self.list_IDs[idx])
        real_audio = self.load_audio(real_audio_file)

        # Augmented real samples as positive data
        augmented_audios = []
        for augment in self.augmentation_methods:
            augmented_audio = globals()[augment](real_audio, self.args, self.sample_rate, audio_path = real_audio_file)
            # print("aug audio shape",augmented_audio.shape)
            augmented_audios.append(np.expand_dims(augmented_audio, axis=1))

        # Additional real audio samples as positive data
        idxs = list(range(len(self.list_IDs)))
        idxs.remove(idx)  # remove the current audio index
        additional_idxs = np.random.choice(idxs, self.num_additional_real, replace=False)
        additional_audios = [np.expand_dims(self.load_audio(os.path.join(self.bonafide_dir, self.list_IDs[i])),axis=1) for i in additional_idxs]
        
        # augment the additional real samples
        augmented_additional_audios = []
        for i in range(len(additional_audios)):
            augmethod_index = random.choice(range(len(self.augmentation_methods)))
            tmp = np.expand_dims(globals()[self.augmentation_methods[augmethod_index]](np.squeeze(additional_audios[i],axis=1), self.args, self.sample_rate, audio_path = os.path.join(self.bonafide_dir, self.list_IDs[additional_idxs[i]])),axis=1)
            augmented_additional_audios.append(tmp)
            
        
        # Additional spoof audio samples as negative data
        additional_spoof_idxs = np.random.choice(self.spoof_list, self.num_additional_spoof, replace=False)
        additional_spoofs = [np.expand_dims(self.load_audio(os.path.join(self.spoof_dir, i)),axis=1) for i in additional_spoof_idxs]
        
        # augment the additional spoof samples
        augmented_additional_spoofs = []
        for i in range(len(additional_spoofs)):
            augmethod_index = random.choice(range(len(self.augmentation_methods)))
            tmp = np.expand_dims(globals()[self.augmentation_methods[augmethod_index]](np.squeeze(additional_spoofs[i],axis=1), self.args, self.sample_rate, audio_path = os.path.join(self.spoof_dir, additional_spoof_idxs[i])),axis=1)
            augmented_additional_spoofs.append(tmp)
        
        # merge all the data
        batch_data = [np.expand_dims(real_audio, axis=1)] + augmented_audios + additional_audios + augmented_additional_audios + additional_spoofs + augmented_additional_spoofs
        batch_data = nii_wav_aug.batch_pad_for_multiview(
                batch_data, self.sample_rate, self.trim_length, random_trim_nosil=True, repeat_pad=self.repeat_pad)
        batch_data = np.concatenate(batch_data, axis=1)
        # print("batch_data.shape", batch_data.shape)
        
        # return will be anchor ID, batch data and label
        batch_data = Tensor(batch_data)
        # label is 1 for anchor and positive, 0 for vocoded
        label = [1] * (len(augmented_audios) +len(additional_audios) + len(augmented_additional_audios) + 1) + [0] * (len(additional_spoofs) + len(augmented_additional_spoofs))
        # print("label", label)
        return self.list_IDs[idx], batch_data, Tensor(label)

class Dataset_for_eval(Dataset):
    def __init__(self, list_IDs, base_dir, padding_type="zero"):
        self.list_IDs = list_IDs
        self.base_dir = os.path.join(base_dir, 'eval')
        self.cut=64600 # take ~4 sec audio (64600 samples)
        self.padding_type = padding_type
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        X_pad = pad(X,self.padding_type,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id

