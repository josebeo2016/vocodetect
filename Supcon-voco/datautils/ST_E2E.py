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
            utt = line.strip().split()
            file_list.append(utt[0])
            d_meta[utt[0]] = 1
        return d_meta,file_list
    
    if (is_dev):
        for line in l_meta:
            utt = line.strip().split()
            file_list.append(utt[0])
            d_meta[utt[0]] = 1
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            utt = line.strip().split(',')[0]
            file_list.append(utt)
        return d_meta,file_list


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
        
        # num_additional_real: number of real samples to add to the batch
        # num_additional_spoof: number of spoof samples to add to the batch
        # the list_IDs will be re-calculated based on the number of real sample in the batch.
        # each element in the list_IDs will be the set of real samples and spoof samples
        
        num_batches = len(list_IDs)//(num_additional_real)
        for i in range(num_batches):
            # randomly select num_additional_real samples from the list_IDs
            idxs = np.random.choice(list_IDs, num_additional_real, replace=False)
            # remove the selected samples from the list_IDs
            for idx in idxs:
                list_IDs.remove(idx)
            # add the selected samples to the list_IDs
            self.list_IDs.append(idxs)
        random.shuffle(self.list_IDs)
        self.list_IDs = self.list_IDs[:int(ratio*len(self.list_IDs))]
        # read spoof_train and spoof_dev list from scp

        self.spoof_train = pd.read_csv(os.path.join(base_dir, 'scp/spoof_train.lst'), sep=",", header=None, names=['path', 'acoustic', 'vocoder', 'vocoder2', 'e2e'])
        self.spoof_dev = pd.read_csv(os.path.join(base_dir, 'scp/spoof_dev.lst'), sep=",", header=None, names=['path', 'acoustic', 'vocoder', 'vocoder2', 'e2e'])
        
        # filter unknown classes
        self.spoof_train = self.spoof_train[self.spoof_train[spoof_category] != 'unknown']
        self.spoof_dev = self.spoof_dev[self.spoof_dev[spoof_category] != 'unknown']
        
        if is_train:
            self.spoof_df = self.spoof_train
        else:
            self.spoof_df = self.spoof_dev
            
        
        
        # make the spoof label map to number for model training.
        # all_spoof = pd.concat([self.spoof_train, self.spoof_dev])
        self.label_map = {}
        # map the spoof class to 1 -> len(categories)
        classes = self.spoof_train[spoof_category].unique()
        classes = np.sort(classes)
        for i, c in enumerate(classes):
            self.label_map[c] = i+1
        
        if not is_train:
            self.spoof_df = self.spoof_df[self.spoof_df[spoof_category].isin(self.spoof_train[spoof_category].unique())]
        
        # add the label column to the spoof dataframe
        if mode == 'E2E':
            print("Mode: E2E")
            print("label_map:", self.label_map)
            print("Training labels:", self.spoof_train[spoof_category].unique())
            self.spoof_df['label'] = self.spoof_df[spoof_category].map(self.label_map)
        else:
            print("Mode: 2stage")
            self.spoof_df['label'] = 1
        
        # print number of samples each classes
        print("Number of samples in each class:")
        for c in self.spoof_df[spoof_category].unique():
            print(f"{c}: {len(self.spoof_df[self.spoof_df[spoof_category] == c])}")
        
        self.repeat_pad = repeat_pad
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
    
    def __getitem__(self, idx):
        # Real samples
        info = str(self.list_IDs[idx])
        real_audios = []
        augmented_audios = []
        for i in range(len(self.list_IDs[idx])):
            real_audio_file = os.path.join(self.base_dir, self.list_IDs[idx][i])
            real_audio = load_audio(real_audio_file)
            real_audios.append(np.expand_dims(real_audio, axis=1))
            augmethod_index = 0 # using RawBoost for all real samples
            augmented_audio = globals()[self.augmentation_methods[augmethod_index]](real_audio, self.args, self.sample_rate, audio_path = real_audio_file)
            augmented_audios.append(np.expand_dims(augmented_audio, axis=1))
        label = [0] * len(real_audios) + [0] * len(augmented_audios)
        # Additional spoof audio samples as negative data
        additional_spoof_idxs = self.spoof_df.sample(n=self.num_additional_spoof)
        additional_spoofs = []
        for _, row in additional_spoof_idxs.iterrows():
            _label = row['label']
            _path = row['path']
            additional_spoof_file = os.path.join(self.base_dir, _path)
            additional_spoof = load_audio(additional_spoof_file)
            additional_spoofs.append(np.expand_dims(additional_spoof, axis=1))
            augmethod_index = 0 # using RawBoost for all spoof samples
            augmented_audio = globals()[self.augmentation_methods[augmethod_index]](additional_spoof, self.args, self.sample_rate, audio_path = additional_spoof_file)
            additional_spoofs.append(np.expand_dims(augmented_audio, axis=1))
            label += [_label] * 2
            

        # merge all the data
        batch_data = real_audios + augmented_audios + additional_spoofs
        batch_data = nii_wav_aug.batch_pad_for_multiview(
                batch_data, self.sample_rate, self.trim_length, random_trim_nosil=True, repeat_pad=self.repeat_pad)
        batch_data = np.concatenate(batch_data, axis=1)

        batch_data = Tensor(batch_data)
        # print("label", label)
        # print("batch_data", batch_data.shape)
        return info, batch_data, Tensor(label)

    def _BK_getitem__(self, idx):
        # Real samples
        info = str(self.list_IDs[idx])
        real_audios = []
        augmented_audios = []
        for i in range(len(self.list_IDs[idx])):
            real_audio_file = os.path.join(self.base_dir, self.list_IDs[idx][i])
            real_audio = load_audio(real_audio_file)
            real_audios.append(np.expand_dims(real_audio, axis=1))
            augmethod_index = 0 # using RawBoost for all real samples
            augmented_audio = globals()[self.augmentation_methods[augmethod_index]](real_audio, self.args, self.sample_rate, audio_path = real_audio_file)
            augmented_audios.append(np.expand_dims(augmented_audio, axis=1))
        label = [0] * len(real_audios) + [0] * len(augmented_audios)
        # Additional spoof audio samples as negative data
        additional_spoof_idxs = self.spoof_df.sample(n=self.num_additional_spoof)
        additional_spoofs = []
        augmented_additional_spoofs = []
        for _, row in additional_spoof_idxs.iterrows():
            _label = row['label']
            _path = row['path']
            additional_spoof_file = os.path.join(self.base_dir, _path)
            additional_spoof = load_audio(additional_spoof_file)
            additional_spoofs.append(np.expand_dims(additional_spoof, axis=1))
            augmethod_index = 0 # using RawBoost for all spoof samples
            augmented_audio = globals()[self.augmentation_methods[augmethod_index]](additional_spoof, self.args, self.sample_rate, audio_path = additional_spoof_file)
            augmented_additional_spoofs.append(np.expand_dims(augmented_audio, axis=1))
            label += [_label] * 2
            

        # merge all the data
        batch_data = real_audios + augmented_audios + additional_spoofs + augmented_additional_spoofs
        batch_data = nii_wav_aug.batch_pad_for_multiview(
                batch_data, self.sample_rate, self.trim_length, random_trim_nosil=True, repeat_pad=self.repeat_pad)
        batch_data = np.concatenate(batch_data, axis=1)

        batch_data = Tensor(batch_data)
        # print("label", label)
        return info, batch_data, Tensor(label)

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