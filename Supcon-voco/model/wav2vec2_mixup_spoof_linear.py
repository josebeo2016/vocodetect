import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os
from model.loss_metrics import supcon_loss
from .xlsr import SSLModel

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
                 dropout_rate, dropout_flag=True):
        super(BackEnd, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        
        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.in_dim, self.out_dim),
            nn.LeakyReLU(),
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
        
        # average pooling -> (batch, self.out_dim)
        feat_utt = feat_.mean(1)
        
        # output linear 
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt

class HardNegativeMixup():
    def __init__(self, device, mix_alpha=0.6, random_mix=False, tau=0.07, is_norm=True, n_synthetic=2):
        super(HardNegativeMixup, self).__init__()
        self.device = device
        self.mix_alpha = mix_alpha
        self.random_mix = random_mix
        self.tau = tau
        self.is_norm = is_norm
        self.n_synthetic = n_synthetic
        
    def sim_metric_seq(self, mat1, mat2):
        '''
        mat1, mat2: tensor, (batch, frame_num, feat_dim)
        output: tensor, (batch, batch)
        '''
        return torch.bmm(mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
    
    def get_hard_negative_idx(self, feats, labels):
        '''
        feats: tensor, (batch, frame_num, feat_dim)
        labels: tensor, (batch, ) [1, 1, 0, 1, ...] (first element is the anchor)
        
        output the indices of the hard negative samples, ordered by the similarity score to the anchor (first element)
        '''
        # get the similarity matrix
        sim_matrix = torch.div(self.sim_metric_seq(feats, feats), self.tau)
        
        # get the similarity scores of the negative samples to the anchor
        anchor_sim = sim_matrix[0, :]
        # sort the similarity scores in ascending order
        _, idx = torch.sort(anchor_sim)
        # get the negative samples' indices
        idx = idx[labels == labels[0]]
        return idx

    
    def mixup(self, feats, labels):
        '''
        feats: tensor, (batch, frame_num, feat_dim)
        labels: tensor, (batch, ) [1, 1, 0, 1, ...]
        
        output: 
        new_feats: tensor, (batch + n_synthetic, frame_num, feat_dim)
        new_labels: tensor, (batch + n_synthetic, ) [1, 1, 0, 1, ..., 0, 0]
        '''
        device = feats.device
        # check number of negative samples (different from the anchor)
        n_neg_samples = torch.sum(labels != labels[0]).item()
        # print(labels)
        # print('n_neg_samples', n_neg_samples)
        # print('n_synthetic', self.n_synthetic)
        assert n_neg_samples > self.n_synthetic, "Not enough negative samples"
        
        # get the indices of the hard negative samples
        hard_neg_idx = self.get_hard_negative_idx(feats, labels)
        
        # generate the synthetic samples
        synthetic_samples = []
        for i in range(self.n_synthetic):
            if self.random_mix:
                self.alpha = random.uniform(0, 1)
            else:
                self.alpha = self.mix_alpha
            
            # randomly select 2 negative samples to mix from the hard negative samples
            # from 0 to n_synthetic
            # regardless of the pair has been selected before
            idx = random.sample(range(self.n_synthetic), 2)
            # get the two negative samples
            neg_sample1 = feats[hard_neg_idx[idx[0]]]
            neg_sample2 = feats[hard_neg_idx[idx[1]]]
            # print('neg_sample1', neg_sample1.shape)
            # print('neg_sample2', neg_sample2)
            # mix the two negative samples
            synthetic_sample = self.alpha * neg_sample1 + (1 - self.alpha) * neg_sample2
            # print('synthetic_sample', synthetic_sample.shape)
            if self.is_norm:
                # apply l2 normalization
                if (synthetic_sample.shape[1] == 1):
                    # when feature is 1D
                    synthetic_sample = F.normalize(synthetic_sample, dim=0)
                else:
                    synthetic_sample = F.normalize(synthetic_sample, dim=1)
            
            synthetic_samples.append(synthetic_sample)
        
        # concatenate the synthetic samples with the original samples
        synthetic_samples = torch.stack(synthetic_samples)
        new_feats = torch.cat((feats, synthetic_samples), dim=0)
        new_labels = torch.cat((labels, torch.zeros(self.n_synthetic).to(device)))
        # make new_labels to be a tensor of int type
        new_labels = new_labels.type(torch.int64)
        return new_feats, new_labels
        
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
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        self.loss_CE = nn.CrossEntropyLoss()
        self.backend = BackEnd(128, 128, 2, 0.5, False)
        
        
        self.harder_mixup = HardNegativeMixup(self.device, mix_alpha=args['mixup']['mix_alpha'], 
                                   random_mix=args['mixup']['random_mix'], tau=args['mixup']['tau'], 
                                   is_norm=args['mixup']['is_norm'], n_synthetic=args['mixup']['n_synthetic'])
        
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        # Post-processing
        
    def _forward(self, x, labels=None):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        if self.flag_fix_ssl:
            with torch.no_grad():
                x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1), is_train = False)
        else:
            x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1), is_train = self.is_train) #(bs,frame_number,feat_dim)
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        if self.is_train:
            # for mixup
            x, labels = self.harder_mixup.mixup(x, labels)
            
        feats = x
        x = nn.ReLU()(x)
        
        # output [batch, 2]
        # emb [batch, 128]
        output, emb = self.backend(x)
        output = F.log_softmax(output, dim=1)
        if (self.is_train):
            return output, feats, emb, labels
        return output
    
    def forward(self, x_big, labels=None):
        # make labels to be a tensor of [bz]
        # labels = labels.squeeze(0)

        if (self.is_train):
            # x_big is a tensor of [1, length, bz]
            # convert to [bz, length]
            # x_big = x_big.squeeze(0).transpose(0,1)
            output, feats, emb, labels = self._forward(x_big, labels)
            # calculate the loss
            return output, feats, emb, labels
        else:
            # in inference mode, we don't need the emb
            # the x_big now is a tensor of [bz, length]
            # print("Inference mode")
            
            return self._forward(x_big)
        
    
    def loss(self, output, feats, emb, labels, config, info=None):
        
        real_bzs = output.shape[0]
        n_views = 1.0
        loss_CE = torch.nn.CrossEntropyLoss()
        
        sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        
        # print("output.shape", output.shape)
        # print("labels.shape", labels.shape)
        L_CE = 1/real_bzs *loss_CE(output, labels)
        
        # reshape the feats to match the supcon loss format
        feats = feats.unsqueeze(1)
        # print("feats.shape", feats.shape)
        L_CF1 = 1/real_bzs * supcon_loss(feats, labels=labels, contra_mode=config['model']['contra_mode'], sim_metric=sim_metric_seq)
        
        # reshape the emb to match the supcon loss format
        emb = emb.unsqueeze(1)
        emb = emb.unsqueeze(-1)
        # print("emb.shape", emb.shape)
        L_CF2 = 1/real_bzs *supcon_loss(emb, labels=labels, contra_mode=config['model']['contra_mode'], sim_metric=sim_metric_seq)
        
        if config['model']['loss_type'] == 1:
            return {'L_CE':L_CE, 'L_CF1':L_CF1, 'L_CF2':L_CF2}
        elif config['model']['loss_type'] == 2:
            return {'L_CE':L_CE, 'L_CF1':L_CF1}
        elif config['model']['loss_type'] == 3:
            return {'L_CE':L_CE, 'L_CF2':L_CF2}
        # ablation study
        elif config['model']['loss_type'] == 4:
            return {'L_CE':L_CE}
        elif config['model']['loss_type'] == 5:
            return {'L_CF1':L_CF1, 'L_CF2':L_CF2}
        
    