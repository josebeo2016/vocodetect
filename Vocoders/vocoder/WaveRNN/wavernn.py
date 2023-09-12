from .inference_only import WaveRNNModel
import torch
import os
from torch import Tensor
from .utils import hparams
import numpy as np

class WaveRNN:
    def __init__(self, checkpoint, device='cuda', config='hparams.py'):
        checkpoint_path = os.path.expanduser(checkpoint)
        
        self.hp = hparams
        self.hp.configure(config)
        
        ckpt = torch.load(checkpoint_path)
        
        print('\nInitialising Model...\n')
        self.device = device

        self.model = WaveRNNModel(rnn_dims=self.hp.voc_rnn_dims,
                        fc_dims=self.hp.voc_fc_dims,
                        bits=self.hp.bits,
                        pad=self.hp.voc_pad,
                        upsample_factors=self.hp.voc_upsample_factors,
                        feat_dims=self.hp.num_mels,
                        compute_dims=self.hp.voc_compute_dims,
                        res_out_dims=self.hp.voc_res_out_dims,
                        res_blocks=self.hp.voc_res_blocks,
                        hop_length=self.hp.hop_length,
                        sample_rate=self.hp.sample_rate,
                        mode=self.hp.voc_mode).to(self.device)
        
        
        self.model.load_state_dict(ckpt)
        
    @torch.no_grad()
    def synthesize(self,mel:np.ndarray):
        # convert to tensor
        mel = Tensor(mel)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)

        audio = self.model.generate_nofile(mel, batched=False, target=None, overlap=None, mu_law=self.hp.mu_law)
        # audio = audio.cpu().detach().numpy()
        return audio
    def __call__(self,mel):
        return self.synthesize(mel)