import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os
from torch.nn.modules.transformer import _get_clones

try:
    from model.loss_metrics import supcon_loss
    from model.RawNet3.model import RawNet3
    from model.RawNet3.RawNetBasicBlock import Bottle2neck
    from model.conformer import ConformerBlock
except:
    from .loss_metrics import supcon_loss
    from .RawNet3.model import RawNet3
    from .RawNet3.RawNetBasicBlock import Bottle2neck
    from .conformer import ConformerBlock
___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

############################
## Conformer model
############################
class MyConformer(nn.Module):
    def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1, pooling='mean'):
        super(MyConformer, self).__init__()
        self.pooling=pooling
        self.dim_head=int(emb_size/heads)
        self.dim=emb_size
        self.heads=heads
        self.kernel_size=kernel_size
        self.n_encoders=n_encoders
        self.encoder_blocks=_get_clones(ConformerBlock(dim = emb_size, dim_head=self.dim_head, heads= heads, 
                ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size),
            n_encoders)
        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)
    
    def forward(self, x): # x shape [bs, tiempo, frecuencia]
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
        for layer in self.encoder_blocks:
            x = layer(x) #[bs,1+tiempo,emb_size]
        if self.pooling=='mean':
            embedding = x.mean(dim=1)
        elif self.pooling=='max':
            embedding = x.max(dim=1)[0]
        else:
            # first token
            embedding=x[:,0,:] #[bs, emb_size]
        out=self.fc5(embedding) #[bs,2]
        return out, embedding


class Model(nn.Module):
    def __init__(self, args, device, is_train = True):
        super().__init__()
        self.device = device
        self.is_train = is_train
        self.flag_fix_ssl = args['flag_fix_ssl']
        self.contra_mode = args['contra_mode']
        self.loss_type = args['loss_type']
        
        self.front_end = RawNet3(
            Bottle2neck,
            model_scale=8,
            context=True,
            summed=True,
            encoder_type="ECA",
            nOut=256,
            out_bn=False,
            sinc_stride=10,
            log_sinc=True,
            norm_sinc="mean",
            grad_mult=1,
        ).to(device)
        
        self.LL = nn.Linear(self.front_end.out_dim, args['conformer']['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        self.backend=MyConformer(**args['conformer'])
        self.loss_CE = nn.CrossEntropyLoss()
        
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        # Post-processing
        
    def _forward(self, x):
        #-----------------RawNet3-----------------#
        x, w = self.front_end(x) #(bs,frame_number,frontend_out_dim)
        x = self.LL(x) #(bs,frame_number,feat_out_dim)
        
        feats = x
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)

        # output [batch, 2]
        # emb [batch, emb_size]
        output, emb = self.backend(x)
        output = F.log_softmax(output, dim=1)
        if (self.is_train):
            return output, feats, emb
        return output
    
    def forward(self, x_big):
        # make labels to be a tensor of [bz]
        # labels = labels.squeeze(0)
        if (x_big.dim() == 3):
            x_big = x_big.transpose(0,1)
            batch, length, sample_per_batch = x_big.shape
            # x_big is a tensor of [length, batch, sample per batch]
            # transform to [length, batch*sample per batch] by concat last dim
            x_big = x_big.transpose(1,2)
            x_big = x_big.reshape(batch * sample_per_batch, length)
        if (self.is_train):
            # x_big is a tensor of [1, length, bz]
            # convert to [bz, length]
            # x_big = x_big.squeeze(0).transpose(0,1)
            output, feats, emb = self._forward(x_big)
            # calculate the loss
            return output, feats, emb
        else:
            # in inference mode, we don't need the emb
            # the x_big now is a tensor of [bz, length]
            print("Inference mode")
            
            return self._forward(x_big)
        
    
    def loss(self, output, feats, emb, labels, config, info=None):
        real_bzs = output.shape[0]
        L_CE = 1/real_bzs *self.loss_CE(output, labels)
        
        # reshape the feats to match the supcon loss format
        feats = feats.unsqueeze(1)
        # print("feats.shape", feats.shape)
        L_CF1 = 1/real_bzs *supcon_loss(feats, labels=labels, contra_mode=self.contra_mode, sim_metric=self.sim_metric_seq)
        
        # reshape the emb to match the supcon loss format
        emb = emb.unsqueeze(1)
        emb = emb.unsqueeze(-1)
        # print("emb.shape", emb.shape)
        L_CF2 = 1/real_bzs *supcon_loss(emb, labels=labels, contra_mode=self.contra_mode, sim_metric=self.sim_metric_seq)
        if self.loss_type == 1:
            return {'L_CE':L_CE, 'L_CF1':L_CF1, 'L_CF2':L_CF2}
        elif self.loss_type == 2:
            return {'L_CE':L_CE, 'L_CF1':L_CF1}
        elif self.loss_type == 3:
            return {'L_CE':L_CE, 'L_CF2':L_CF2}
        # ablation study
        elif self.loss_type == 4:
            return {'L_CE':L_CE}
        elif self.loss_type == 5:
            return {'L_CF1':L_CF1, 'L_CF2':L_CF2}
        
