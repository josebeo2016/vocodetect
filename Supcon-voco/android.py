import os
from re import S
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile
import typing
import time

from model.wav2vec2_linear_nll_torchaudio import Model
# from model.wav2vec2_linear_nll import Model
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.is_train=False
        self.model.eval()
        self.min_score = -8.0
        self.max_score = 0.0
        self.threshold = -2.8
    def scaled_likelihood(self, score:float) -> float:
        '''
        if score =< threshold, then scale it to [0, 50)
        if score > threshold, then scale it to [50, 100]
        based on the min_score and max_score
        '''
        if score <= self.threshold:
            scaled = (score - self.min_score) / (self.threshold - self.min_score) * 50
        else:
            scaled =  50 + (score - self.threshold) / (self.max_score - self.threshold) * 50
        
        if scaled < 0:
            scaled = 0
        elif scaled > 100:
            scaled = 100
        return scaled
    def pad(self, x, max_len: int = 64600):
        x_len = x.shape[0]
        # print(x_len)
        if (x_len>=max_len):
            pad_x =  x[:max_len]
        else:
            num_repeats = int(max_len/x_len)+1
            pad_x = x.repeat(1,num_repeats)[0][:max_len]
        # print(pad_x)
        return pad_x.unsqueeze(0)
    
    def forward(self, wavforms: Tensor):
        wav_padded = self.pad(wavforms)
        # print(wav_padded.shape)
        out = self.model(wav_padded)
        # convert to probability
        scaled_likelihood = self.scaled_likelihood(out[0][1].item())
        fake_prob = 100-scaled_likelihood
        return fake_prob

config = yaml.load(open("configs/5_augall_wav2vec2_linear_nll_eval_only.yaml", 'r'), Loader=yaml.FullLoader)
model_path = "out/model_weighted_CCE_100_1_1e-08_5_augall_wav2vec2_linear_supcon_jan22_from_nov22_r_29/epoch_36.pth"
device='cpu'
model = Model(config['model'], device)
# fix state dict missing and unexpected keys
pretrained_dict = torch.load(model_path)
pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
pretrained_dict = {key.replace("_orig_mod.", ""): value for key, value in pretrained_dict.items()}
model.load_state_dict(pretrained_dict)

_model = ModelWrapper(model)
# Sanity check
# check running time
start_time = time.time()
file_path = "/dataa/phucdt/vocodetect/traindata/intern_2024/2024_FakeSample/TTS_Sample/TTS_arab_9.wav"
data, y = librosa.load(file_path, sr=16000)
res = _model(Tensor(data))
print("Fake: {}%".format(res))
print("Time: ", time.time()-start_time)


# # Apply quantization / script / optimize for motbile
_model.eval()
print("DEBUG2")
scripted_model = torch.jit.script(_model)
print("DEBUG3")
optimized_model = optimize_for_mobile(scripted_model)
print("DEBUG4")

# Sanity check
start_time = time.time()
file_path = "/dataa/phucdt/vocodetect/traindata/intern_2024/2024_FakeSample/TTS_Sample/TTS_arab_9.wav"
data, y = librosa.load(file_path, sr=16000, mono=True)
print(Tensor(data))
res = optimized_model(Tensor(data))
print("DEBUG5")
print("Fake: {}%".format(res))
print("Time: {:2f}s", time.time()-start_time)

# print('Result:', optimized_model(Tensor(data)))
# optimized_model._save_for_lite_interpreter("btsdetect_cnn.ptl")