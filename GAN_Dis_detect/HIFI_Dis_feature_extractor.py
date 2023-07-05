from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
from unittest import result
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from hifigan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from utils import load_checkpoint
import random
import soundfile as sf
MAX_WAV_VALUE = 32768.0
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_mel(x,sr,h):
    return mel_spectrogram(y=x, n_fft=h.n_fft, num_mels=h.num_mels, sampling_rate = sr, hop_size=h.hop_size, win_size=h.win_size, fmin=h.fmin, fmax=sr/2.0)

def load_flac(full_path):
    data, sampling_rate = sf.read(full_path)
    return data, sampling_rate


def process_wav(in_path, out_path):
    
    wav, sr = load_flac(in_path)
    wav = torch.FloatTensor(wav).to(device)
    wav = wav.unsqueeze(0)
    
    if wav.size(1) >= h.segment_size:
        max_audio_start = wav.size(1) - h.segment_size
        audio_start = random.randint(0, max_audio_start)
        wav = wav[:, audio_start:audio_start+h.segment_size]
    else:
        wav = torch.nn.functional.pad(wav, (0, h.segment_size - wav.size(1)), 'constant')

    wav = wav.squeeze(0)
    wav = wav.unsqueeze(0).unsqueeze(1)
    # OLD
    _, _, fmap_f_r, _ = mpd(wav,wav.detach())
    _, _, fmap_s_r, _ = msd(wav,wav.detach())
    emb_d = torch.flatten(fmap_f_r[-1][-1], 1, -1).detach().squeeze()
    emb_s = torch.flatten(fmap_s_r[-1][-1], 1, -1).detach().squeeze()
    x = torch.cat((emb_d, emb_s),0)
    
    #NEW
    # y_d_rs, y_d_gs, fmap_d_rs, fmap_d_gs = mpd(wav,wav.detach())
    # y_s_rs, y_s_gs, fmap_s_rs, fmap_s_gs = msd(wav,wav.detach())
    
    # tmp = torch.Tensor().to(device)
    # max_len = len(y_d_rs[0][0])
    # # print(max_len)
    # for i in y_d_rs:
    #     # print(i[0,:max_len].shape)
    #     tmp=torch.cat((tmp,i[0,:max_len]),dim=-1)
    
    # y_d_rs[0][0].detach().cpu().shape
    # tmp=tmp.reshape((5,max_len))
    # OLD
    torch.save(x.detach().cpu(), out_path)
    # #NEW
    # torch.save(tmp.detach().cpu(), out_path)
    # print("DEBUG {} {}".format(in_path,out_path))
    return out_path, -1


def preprocess_dataset(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    path = []
    executor = ProcessPoolExecutor(max_workers=4)
    print(f"Resampling audio in {args.in_dir}")
    for in_path in tqdm(args.in_dir.rglob("*.flac")):
        relative_path = in_path.relative_to(args.in_dir)
        out_path = args.out_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.append(process_wav(in_path, out_path))

    print("Finish extract hifi-gan feature for {} audio".format(len(results),results[0][-1]))


config_file = "/dataa/phucdt/vocodetect/hifi-gan/cp_16k/config.json"
do_file = "/dataa/phucdt/vocodetect/hifi-gan/cp_16k/do_00255000"
g_file = "/dataa/phucdt/vocodetect/hifi-gan/cp_16k/g_00255000"
device = "cuda:1"

with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

generator = Generator(h).to(device)

state_dict_g = load_checkpoint(g_file, device)
generator.load_state_dict(state_dict_g['generator'])

mpd = MultiPeriodDiscriminator().to(device)
msd = MultiScaleDiscriminator().to(device)

state_dict_do = load_checkpoint(do_file, device)

mpd.load_state_dict(state_dict_do['mpd'])
msd.load_state_dict(state_dict_do['msd'])

mpd.eval()
msd.eval()
    

parser = argparse.ArgumentParser(description="Resample an audio dataset.")
parser.add_argument(
    "in_dir", metavar="in-dir", help="path to the dataset directory.", type=Path
)
parser.add_argument(
    "out_dir", metavar="out-dir", help="path to the output directory.", type=Path
)

args = parser.parse_args()
preprocess_dataset(args)