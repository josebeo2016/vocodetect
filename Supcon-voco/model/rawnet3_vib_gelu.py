import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os
try:
    from model.loss_metrics import supcon_loss
    from model.RawNet3 import RawNet3
    from model.RawNet3.RawNetBasicBlock import Bottle2neck
except:
    from loss_metrics import supcon_loss
    from .RawNet3.RawNetBasicBlock import Bottle2neck
    from .RawNet3.model import RawNet3


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

class DropoutForMC(nn.Module):
    """Dropout layer for Bayesian model
    THe difference is that we do dropout even in eval stage
    """
    def __init__(self, p, dropout_flag=True):
        super(DropoutForMC, self).__init__()
        self.p = p
        self.flag = dropout_flag
        return
        
    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, training=self.flag)

class BackEnd(nn.Module):
    """Back End Wrapper
    """
    def __init__(self, input_dim, out_dim, num_classes, 
                 dropout_rate, dropout_flag=True, pooling='mean'):
        super(BackEnd, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes
        # pooling
        self.pooling = pooling
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        
        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate)
        )
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag))

        # linear layer to produce output logits 
        self.m_utt_level = nn.Linear(self.out_dim, self.num_class)
        
        return

    def forward(self, feat):
        """ logits, emb_vec = back_end_emb(feat)

        input:
        ------
          feat: tensor, (batch, frame_num, feat_feat_dim)

        output:
        -------
          logits: tensor, (batch, num_output_class)
          emb_vec: tensor, (batch, emb_dim)
        
        """
        # through the frame-level network
        # (batch, frame_num, self.out_dim)
        feat_ = self.m_frame_level(feat)
        if (self.pooling=='mean'):
            # average pooling -> (batch, self.out_dim)
            feat_utt = feat_.mean(1)
        else:
            # max pooling -> (batch, self.out_dim)
            feat_utt = feat_.max(1)[0]
        # average pooling -> (batch, self.out_dim)
        # feat_utt = feat_.mean(1)
        
        # max pooling -> (batch, self.out_dim)
        feat_utt = feat_.max(1)[0]
        
        # output linear 
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt
    

class VIB(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, m_mcdp_rate=0.5, mcdp_flag=True):
        super(VIB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.m_mcdp_rate = m_mcdp_rate
        self.m_mcdp_flag = mcdp_flag

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
        )

        # Latent space
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = self.fc_mu(encoded), self.fc_var(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return z, decoded, mu, logvar

class Model(nn.Module):
    def __init__(self, args, device, is_train = True):
        super().__init__()
        self.device = device
        self.is_train = is_train
        self.flag_fix_ssl = args['flag_fix_ssl']
        self.contra_mode = args['contra_mode']
        self.loss_type = args['loss_type']
        ####
        # create network wav2vec 2.0
        ####
        # self.ssl_model = SSLModel(self.device)
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
        
        self.LL = nn.Linear(self.front_end.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.is_freeze_frontend = False if 'is_freeze_frontend' not in args else args['is_freeze_frontend']
        
        self.selu = nn.SELU(inplace=True)
        nclasses = args['nclasses'] if 'nclasses' in args else 2
        self.loss_CE = nn.CrossEntropyLoss()
        self.VIB = VIB(128, 128, 64)
        self.backend = BackEnd(64, 64, nclasses, 0.5, False)
        
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        # Post-processing
        if self.is_freeze_frontend:
            self.freeze_layers()
            self.backend_stage2 = BackEnd(64, 64, nclasses, 0.5, False)
    def freeze_layers(self):
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze the backend parameters
        for param in self.backend.parameters():
            param.requires_grad = True
                
    def _forward(self, x):
        #-----------------RawNet3-----------------#
        x, w = self.front_end(x) #(bs,frame_number,frontend_out_dim)

        x = self.LL(x) #(bs,frame_number,feat_out_dim)
        feats = x
        x = nn.GELU()(x)
        
        # VIB
        # x [batch, frame_number, 64]
        x, decoded, mu, logvar = self.VIB(x)
        
        # output [batch, 2]
        # emb [batch, 64]
        output, emb = self.backend(x)
        # output = F.log_softmax(output, dim=1)
        output = F.softmax(output, dim=1)
        if (self.is_train):
            return output, (decoded, mu, logvar, feats), emb
        return output
    
    def _forward_freeze(self, x):
        #-----------------RawNet3-----------------#
        # freeze the front end
        with torch.no_grad():
            x, w = self.front_end(x) #(bs,frame_number,frontend_out_dim)
            x = self.LL(x) #(bs,frame_number,feat_out_dim)
            feats = x
            x = nn.GELU()(x)
            
            # VIB
            # x [batch, frame_number, 64]
            x, decoded, mu, logvar = self.VIB(x)
        
        # output [batch, 2]
        # emb [batch, 64]
        if self.is_freeze_frontend:
            output, emb = self.backend_stage2(x)
        else:
            output, emb = self.backend(x)
        # output = F.log_softmax(output, dim=1)
        output = F.softmax(output, dim=1)
        if (self.is_train):
            return output, (decoded, mu, logvar, feats), emb
        return output
    
    def forward(self, x_big):
        # make labels to be a tensor of [bz]
        # labels = labels.squeeze(0)

        if (self.is_train):
            # x_big is a tensor of [1, length, bz]
            # convert to [bz, length]
            # x_big = x_big.squeeze(0).transpose(0,1)
            if self.is_freeze_frontend:
                return self._forward_freeze(x_big)
            else:
                return self._forward(x_big)
        else:
            # in inference mode, we don't need the emb
            # the x_big now is a tensor of [bz, length]
            # print("Inference mode")
            return self._forward(x_big)
    
    # def forward(self, x_big):
    #     # make labels to be a tensor of [bz]
    #     # labels = labels.squeeze(0)
    #     if (x_big.dim() == 3):
    #         x_big = x_big.transpose(0,1)
    #         batch, length, sample_per_batch = x_big.shape
    #         # x_big is a tensor of [length, batch, sample per batch]
    #         # transform to [length, batch*sample per batch] by concat last dim
    #         x_big = x_big.transpose(1,2)
    #         x_big = x_big.reshape(batch * sample_per_batch, length)
    #     if (self.is_train):
    #         # x_big is a tensor of [1, length, bz]
    #         # convert to [bz, length]
    #         # x_big = x_big.squeeze(0).transpose(0,1)
    #         output, feats, emb = self._forward(x_big)
    #         # calculate the loss
    #         return output, feats, emb
    #     else:
    #         # in inference mode, we don't need the emb
    #         # the x_big now is a tensor of [bz, length]
    #         # print("Inference mode")
    #         return self._forward(x_big)
        
    
    def loss(self, output, feats, emb, labels, config, info=None):
        '''
        output: tensor, (batch, num_output_class)
        feats: tuple: feats[0] decoded (batch, frame_num, feat_feat_dim)
                      feats[1] mu (batch, frame_num, feat_feat_dim)
                      feats[2] logvar (batch, frame_num, feat_feat_dim)
                      feats[3] wav2vec feats (batch, frame_num, feat_feat_dim)
        emb: tensor, (batch, emb_dim)
        '''
        
        # get loss weights from config, default is 1.0
        weight_CE = config['model']['weight_CE'] if 'weight_CE' in config['model'] else 1.0
        weight_CF1 = config['model']['weight_CF1'] if 'weight_CF1' in config['model'] else 1.0
        weight_CF2 = config['model']['weight_CF2'] if 'weight_CF2' in config['model'] else 1.0
        recon_weight_l = config['model']['recon_weight_l'] if 'recon_weight_l' in config['model'] else 0.000001
        recon_weight_b = config['model']['recon_weight_b'] if 'recon_weight_b' in config['model'] else 0.05
        
        real_bzs = output.shape[0]
        n_views = 1.0
        loss_CE = torch.nn.CrossEntropyLoss()
        if config['model']['loss_type'] == 4:
            L_CE = weight_CE * 1/real_bzs *loss_CE(output, labels)
            return {'L_CE':L_CE}
        
        decoded, mu, logvar, feats_w2v = feats
        
        sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        
        # print("output.shape", output.shape)
        # print("labels.shape", labels.shape)
        L_CE = weight_CE * 1/real_bzs *loss_CE(output, labels)
        
        # Recon loss
        # print("decoded: ", decoded.shape)
        # print("feats_w2v: ", feats_w2v.shape)
        # print("mu: ", mu.shape)
        # print("logvar: ", logvar.shape)

        BCE = F.binary_cross_entropy(torch.sigmoid(decoded), torch.sigmoid(feats_w2v), reduction='sum')
        # print("BCE: ", BCE)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print("KLD: ", KLD)
        # Recon_loss = 0.000001*(BCE + 0.05*KLD) / real_bzs
        Recon_loss = recon_weight_l*(BCE + recon_weight_b*KLD) / real_bzs
        # reshape the feats_w2v to match the supcon loss format
        feats_w2v = feats_w2v.unsqueeze(1)
        # print("feats_w2v.shape", feats_w2v.shape)
        L_CF1 = weight_CF1* 1/real_bzs * supcon_loss(feats_w2v, labels=labels, contra_mode=config['model']['contra_mode'], sim_metric=sim_metric_seq)
        
        # reshape the emb to match the supcon loss format
        emb = emb.unsqueeze(1)
        emb = emb.unsqueeze(-1)
        # print("emb.shape", emb.shape)
        L_CF2 = weight_CF2* 1/real_bzs *supcon_loss(emb, labels=labels, contra_mode=config['model']['contra_mode'], sim_metric=sim_metric_seq)
        
        if config['model']['loss_type'] == 1:
            return {'L_CE':L_CE, 'L_CF1':L_CF1, 'L_CF2':L_CF2, 'Recon_loss':Recon_loss}
        elif config['model']['loss_type'] == 2:
            return {'L_CE':L_CE, 'L_CF1':L_CF1}
        elif config['model']['loss_type'] == 3:
            return {'L_CE':L_CE, 'L_CF2':L_CF2}
        # ablation study
        elif config['model']['loss_type'] == 4:
            return {'L_CE':L_CE, 'Recon_loss':Recon_loss}
        elif config['model']['loss_type'] == 5:
            return {'L_CF1':L_CF1, 'L_CF2':L_CF2}
        
