import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MLP(nn.Module):
    """MLP backend
    """
    def __init__(self, input_dim, out_dim, num_classes, 
                  hidden_dim=[64,64], dropout_rate=0.5, dropout_flag=True):
        super(MLP, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes
        # hidden dimension
        self.hidden_dim = hidden_dim
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        
        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim[0]),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.hidden_dim[0], self.hidden_dim[1]),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.hidden_dim[1], self.out_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate)
        )
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag))

        # linear layer to produce output logits 
        self.m_utt_level = nn.Linear(self.out_dim, self.num_class)
        
        return
    # def initialize_parameters(self):
    #     """Randomly initialize the parameters of the model."""
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='gelu')
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)
                    
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
        
        # print(feat.shape)
        # change from 
        feat_ = self.m_frame_level(feat)
        
        # average pooling -> (batch, self.out_dim)
        # print("feat_.shape", feat_.shape)
        if len(feat_.shape) == 3:
            # feat_utt = torch.mean(feat_, dim=1) # average pooling over the frames
            # max pooling
            feat_utt = torch.max(feat_, dim=1)[0]
        else:
            feat_utt = feat_
        # output linear 
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt
    
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
        nclasses = args['nclasses'] if 'nclasses' in args else 2
        self.backend = MLP(input_dim = args['mlp']['input_dim'], out_dim = args['mlp']['out_dim'], 
                           num_classes = nclasses, hidden_dim = args['mlp']['hidden_dim'], dropout_rate=0.5, dropout_flag=False)
        
    def _forward(self, x):
        output, emb = self.backend(x)
        # output = F.log_softmax(output, dim=1)
        # print("output.shape", output.shape)
        output = torch.softmax(output, dim=1)
        
        if (self.is_train):
            return output, emb, emb
        return output
        
    def forward(self, x_big):
        # make labels to be a tensor of [bz]
        # labels = labels.squeeze(0)
        # print("x_big.shape", x_big.shape)
        # change from (length, batch, feat_size) to (batch, length, feat_size)
        if len(x_big.shape) == 3:
            x_big = x_big.permute(1,0,2)

        if (self.is_train):
            return self._forward(x_big)
        else:
            # in inference mode, we don't need the emb
            # the x_big now is a tensor of [bz, length]
            # print("Inference mode")
            return self._forward(x_big)
        
    
    def loss(self, output, feats, emb, labels, config, info=None):
        '''
        output: tensor, (batch, num_output_class)
        feats: tuple: feats[0] decoded (batch, frame_num, feat_feat_dim)
                      feats[1] mu (batch, frame_num, feat_feat_dim)
                      feats[2] logvar (batch, frame_num, feat_feat_dim)
                      feats[3] wav2vec feats (batch, frame_num, feat_feat_dim)
        emb: tensor, (batch, emb_dim)
        '''
        batch_size = output.shape[0]
        n_views = output.shape[1]
        real_bzs = batch_size * n_views
        
        loss_CE = torch.nn.CrossEntropyLoss()
        # get loss weights from config, default is 1.0
        
        weight_CE = config['model']['weight_CE'] if 'weight_CE' in config['model'] else 1.0
        if len(output.shape) == 3:
            CE_labels = labels.repeat(n_views) # repeat the labels for CE loss
            CE_output = output.view(real_bzs, -1)
            #
            # print(f"CE_output.shape: {CE_output.shape}, CE_labels.shape: {CE_labels.shape}")
            L_CE = weight_CE * 1/real_bzs *loss_CE(CE_output, CE_labels)
        else:
            # print(output.shape, labels.shape)
            L_CE = weight_CE * 1/batch_size *loss_CE(output, labels)
        return {'L_CE': L_CE}