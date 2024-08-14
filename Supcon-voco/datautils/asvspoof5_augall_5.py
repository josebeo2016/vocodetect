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
from datautils.dataio import pad, load_audio
# augwrapper
from datautils.augwrapper import SUPPORTED_AUGMENTATION

# dynamic import of augmentation methods
for aug in SUPPORTED_AUGMENTATION:
    exec(f"from datautils.augwrapper import {aug}")
    

logging.basicConfig(filename='errors.log', level=logging.DEBUG)
def genList(dir_meta, is_train=False, is_eval=False, is_dev=False):
    # bonafide: 1, spoof: 0
    d_meta = {}
    file_list=[]
    # get dir of metafile only
    dir_meta = os.path.dirname(dir_meta)
    if is_train:
        metafile = os.path.join(dir_meta, 'scp/bonafide_train.lst')
    elif is_dev:
        metafile = os.path.join(dir_meta, 'scp/bonafide_dev.lst')
    elif is_eval:
        metafile = os.path.join(dir_meta, 'scp/eval.lst')
        
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
    

class Dataset_for(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[], 
                 augmentation_methods=[], num_additional_real=2, num_additional_spoof=2, 
                 trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None, 
                 aug_dir=None, online_aug=False, repeat_pad=True, is_train=True):
        """
        Args:
            list_IDs (string): Path to the .lst file with real audio filenames.
            vocoders (list): list of vocoder names.
            augmentation_methods (list): List of augmentation methods to apply.
        """
        self.args = args
        self.args.noise_path = noise_path
        self.args.rir_path = rir_path
        self.args.aug_dir = aug_dir
        self.args.online_aug = online_aug
        self.list_IDs = list_IDs
        # self.bonafide_dir = os.path.join(base_dir, 'bonafide')
        # self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        # # list available spoof samples (only .wav files)
        # self.spoof_dir = os.path.join(base_dir, 'spoof')
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        self.base_dir = base_dir
        
        # read spoof_train and spoof_dev list from scp
        if is_train:
            self.spoof_list = []
            with open(os.path.join(base_dir, 'scp/spoof_train.lst'), 'r') as f:
                self.spoof_list = f.readlines()
            self.spoof_list = [i.strip() for i in self.spoof_list]
        else:
            self.spoof_list = []
            with open(os.path.join(base_dir, 'scp/spoof_dev.lst'), 'r') as f:
                self.spoof_list = f.readlines()
            self.spoof_list = [i.strip() for i in self.spoof_list]
        
        
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

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # Anchor real audio sample
        real_audio_file = os.path.join(self.base_dir, self.list_IDs[idx])
        real_audio = load_audio(real_audio_file)

        # Vocoded audio samples as negative data
        vocoded_audio_files = [os.path.join(self.vocoded_dir, vf, self.list_IDs[idx].split("/")[-1]) for vf in self.vocoders]
        vocoded_audios = []
        augmented_vocoded_audios = []
        for vf in vocoded_audio_files:
            vocoded_audio = load_audio(vf)
            vocoded_audios.append(np.expand_dims(vocoded_audio, axis=1))
            # Augmented vocoded samples as negative data with first augmentation method
            augmented_vocoded_audio = globals()[self.augmentation_methods[0]](vocoded_audio, self.args, self.sample_rate, audio_path = vf)
            augmented_vocoded_audios.append(np.expand_dims(augmented_vocoded_audio, axis=1))
        
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
        additional_audios = [np.expand_dims(load_audio(os.path.join(self.base_dir, self.list_IDs[i])),axis=1) for i in additional_idxs]
        
        # Additional spoof audio samples as negative data
        additional_spoof_idxs = np.random.choice(self.spoof_list, self.num_additional_spoof, replace=False)
        additional_spoofs = [np.expand_dims(load_audio(os.path.join(self.base_dir,i)),axis=1) for i in additional_spoof_idxs]
        
        # augment the additional spoof samples with the copy-paste augmentation method
        augmented_additional_spoofs = []
        for spoof in additional_spoofs:
            augmented_spoof = globals()['copy_paste_r'](spoof, self.args, self.sample_rate, audio_path = real_audio_file)
            augmented_additional_spoofs.append(np.expand_dims(augmented_spoof, axis=1))
        
        # merge all the data
        batch_data = [np.expand_dims(real_audio, axis=1)] + augmented_audios + additional_audios + vocoded_audios + augmented_vocoded_audios + additional_spoofs + augmented_additional_spoofs
        batch_data = nii_wav_aug.batch_pad_for_multiview(
                batch_data, self.sample_rate, self.trim_length, random_trim_nosil=True, repeat_pad=self.repeat_pad)
        batch_data = np.concatenate(batch_data, axis=1)
        # print("batch_data.shape", batch_data.shape)
        
        # return will be anchor ID, batch data and label
        batch_data = Tensor(batch_data)
        # label is 1 for anchor and positive, 0 for vocoded
        label = [1] * (len(augmented_audios) +len(additional_audios) + 1) + [0] * (len(self.vocoders*2) +  len(additional_spoofs) + len(augmented_additional_spoofs))
        return self.list_IDs[idx], batch_data, Tensor(label)

class Dataset_for_eval(Dataset):
    def __init__(self, list_IDs, base_dir, max_len=64600, padding_type="repeat"):
        self.list_IDs = list_IDs
        self.base_dir = os.path.join(base_dir)
        self.trim_length=max_len 
        self.padding_type = padding_type
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
            
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X, _ = librosa.load(filepath, sr=16000)
        X_pad= pad(X,padding_type=self.padding_type,max_len=self.trim_length, random_start=True)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id
