import os
import numpy as np
import torch
import torchaudio
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
import librosa
from core_scripts.data_io import wav_augmentation as nii_wav_aug
from core_scripts.data_io import wav_tools as nii_wav_tools
from datautils.RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import logging
import random
from datautils.dataio import pad, load_audio
# augwrapper
from datautils.augwrapper import SUPPORTED_AUGMENTATION

# dynamic import of augmentation methods
for aug in SUPPORTED_AUGMENTATION:
    exec(f"from datautils.augwrapper import {aug}")
    

#########################
# Set up logging
#########################
import logging
from logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def genList(dir_meta, is_train=False, is_eval=False, is_dev=False):
    # bonafide: 1, spoof: 0
    d_meta = {}
    file_list=[]
    # get dir of metafile only
    dir_meta = os.path.dirname(dir_meta)
    protocol = os.path.join(dir_meta, "meta_ST.csv")
    meta_df = pd.read_csv(protocol)

    if (is_train):
        train_meta = meta_df[meta_df['subset']=='train']
        file_list = train_meta['path'].tolist()
        d_meta = meta_df
        return d_meta, file_list
    
    if (is_dev):
        dev_meta = meta_df[meta_df['subset']=='dev']
        file_list = dev_meta['path'].tolist()
        d_meta = meta_df
        return d_meta, file_list
    
    if(is_eval):
        eval_meta = meta_df[meta_df['subset']=='eval']
        file_list = eval_meta['path'].tolist()
        d_meta = []
        return d_meta, file_list


class Dataset_for(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[], 
                 augmentation_methods=[], num_additional_real=2, num_additional_spoof=2, 
                 trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None, 
                 aug_dir=None, online_aug=False, repeat_pad=True, is_train=True, spoof_category='acoustic', mode='E2E', ratio=1.0):
        """
        Args:
            list_IDs (string): Path to the .lst file with real audio filenames.
            vocoders (list): list of vocoder names.
            augmentation_methods (list): List of augmentation methods to apply.
            spoof_category: acoustic, vocoder, vocoder2
            mode: E2E, 2stage
        """
        self.args = args
        self.args.noise_path = noise_path
        self.args.rir_path = rir_path
        self.args.aug_dir = aug_dir
        self.args.online_aug = online_aug
        self.list_IDs = []
        self.bonafide_dir = os.path.join(base_dir, 'bonafide')
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        self.spoof_dir = os.path.join(base_dir, 'spoof')
        self.base_dir = base_dir
        self.is_train = is_train
        self.mode = mode
        
        meta_df = labels.copy()
        # keep ratio samples
        # meta_df = meta_df.sample(frac=ratio, random_state=1234).reset_index(drop=True)
        # mlaad_df = mlaad_df[mlaad_df['vocoder']!='unknown']
        
        meta_df = meta_df[meta_df[spoof_category]!='unknown']
        seen_classes = meta_df[meta_df['subset']=='train'][spoof_category].unique()
        seen_classes = np.sort(seen_classes)
        if is_train:
            meta_df = meta_df[meta_df['subset']=='train']
        if not is_train:
            meta_df = meta_df[meta_df['subset']=='dev']
            meta_df = meta_df[meta_df[spoof_category].isin(seen_classes)]
        label_map = {seen_classes[i]: i for i in range(len(seen_classes))}
        t = len(label_map)
        classes = meta_df[spoof_category].unique()
        for c in classes:
            if c not in label_map:
                label_map[c] = t
                t += 1
        print(label_map)
        
        # shuffle the data
        meta_df = meta_df.sample(frac=1, random_state=1234).reset_index(drop=True)
        self.list_IDs = meta_df['path'].tolist()
        if self.mode == 'E2E':
            self.labels = meta_df[spoof_category].map(label_map).tolist()
        elif self.mode == '2stage':
            # convert to int
            map_binary = {'bonafide': 0, 'spoof': 1}
            print("2stage mode")
            self.labels = meta_df['label'].map(map_binary).tolist()
        # meta_df['label_map'] = meta_df[spoof_category].map(label_map)
        # self.labels = meta_df['label_map'].tolist()

        self.repeat_pad = repeat_pad
        if self.repeat_pad:
            self.padding_type = "repeat"
        else:
            self.padding_type = "zero"
        self.trim_length = trim_length
        self.sample_rate = wav_samp_rate

        self.vocoders = vocoders
        # print("vocoders:", vocoders)
        self.num_additional_spoof = num_additional_spoof
        self.num_additional_real = num_additional_real
        self.augmentation_methods = augmentation_methods
        
        if len(augmentation_methods) < 1:
            # using default augmentation method RawBoostWrapper12
            self.augmentation_methods = ["RawBoost12"]

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # Real samples
        utt_id = self.list_IDs[index]
        label = self.labels[index]
        audio_path = os.path.join(self.base_dir, utt_id)
        X = load_audio(audio_path)
        X_pad = pad(X, self.padding_type, self.trim_length)
        # augmentations
        augmethod_index = random.randint(0, len(self.augmentation_methods)-1)
        X_pad = globals()[self.augmentation_methods[augmethod_index]](X_pad, self.args, self.sample_rate, audio_path = audio_path)
        x_inp = Tensor(X_pad)
        return utt_id, x_inp, label
        # return info, batch_data, Tensor(label)


class Dataset_for_eval(Dataset):
    # [TODO]
    def __init__(self, list_IDs, base_dir, max_len=64600, padding_type="zero"):
        self.list_IDs = list_IDs
        self.base_dir = os.path.join(base_dir)
        self.cut=max_len 
        self.padding_type = padding_type
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        X_pad = pad(X,self.padding_type,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id