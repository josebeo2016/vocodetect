import glob
import os
import numpy as np
import argparse
import json
import torch
import numpy as np
import os
import torchaudio

from argparse import ArgumentParser

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

class Diff_inference:
    def __init__(self, checkpoint:os.PathLike, h=None, device='cuda'):
        
        checkpoint = os.path.expanduser(checkpoint)
        self.model = DiffWave(AttrDict(base_params)).to(device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()      
    

    def inference(self,x):
        with torch.no_grad():
            if isinstance(x,np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            else:
                x = x.to(self.device)
            y_g_hat = self.generator(x.unsqueeze(0))
            audio = y_g_hat.squeeze()
            audio = audio.cpu().numpy()

        return audio
    def __call__(self,x):
        return self.inference(x)



