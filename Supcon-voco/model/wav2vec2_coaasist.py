
from torch import nn
import torch
import torch.nn.functional as F
from conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones
import random
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import fairseq
import logging
from typing import Optional
try:
    from model.loss_metrics import supcon_loss
    from model.xlsr import SSLModel
except:
    from .loss_metrics import supcon_loss
    from .xlsr import SSLModel



''' Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu and Nicholas Evans. 
    AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks. 
    In Proc. ICASSP 2022, pp: 6367--6371.'''


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master: Optional[torch.Tensor] = None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        # print('num_type1',num_type1)
        # print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        # print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        # print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('Concat x1 and x2',x.shape)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            # print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        # Convert num_type1 to a tensor if it's not already
        if not isinstance(num_type1, torch.Tensor):
            num_type1 = torch.tensor(num_type1)

        # Convert num_type2 to a tensor if it's not already
        if not isinstance(num_type2, torch.Tensor):
            num_type2 = torch.tensor(num_type2)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        # print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        # print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        # print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        # print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        # self.k = k
        self.k = torch.tensor(k)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        # Convert self.k to a tensor if it's not already

        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        # n_nodes = max(int(n_nodes * k), 1)
        # n_nodes = torch.max(torch.tensor([int(n_nodes * k), 1]))
        n_nodes = torch.max(
            (torch.as_tensor(n_nodes) * k).long(), torch.as_tensor(1))
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        self.bn1 = None
        self.conv_downsample = None

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first and self.bn1 is not None:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        # print('out',out.shape)
        out = self.conv1(x)

        # print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out',out.shape)

        if self.downsample and self.conv_downsample is not None:
            identity = self.conv_downsample(identity)

        out += identity
        # out = self.mp(out)
        return out


class Model(nn.Module):
    def __init__(self, args,device, is_train = True):
        super().__init__()
        self.flag_fix_ssl = args['flag_fix_ssl']
        self.contra_mode = args['contra_mode']
        self.loss_type = args['loss_type']
        self.is_train = is_train
        self.emb_size = args['emb_size']
        self.heads = args['heads']
        self.ffmult = args['ffmult']
        self.exp_fac = args['exp_fac']
        self.kernel_size = args['kernel_size']
        self.n_encoders = args['n_encoders']

        self.dim_head = int(self.emb_size/self.heads)

        # AASIST parameters
        filts = args['aasist']['filts']
        gat_dims = args['aasist']['gat_dims']
        pool_ratios = args['aasist']['pool_ratios']
        temperatures =  args['aasist']['temperatures']

        ####
        # create network wav2vec 2.0
        ####

        self.ssl_model = SSLModel(device).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        # Replace the original Residual_block with the one from the AASIST model
        # to ConformerBlock
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        self.encoder_conformer = _get_clones(ConformerBlock(dim=self.emb_size, dim_head=self.dim_head, heads=self.heads,
                                                            ff_mult=self.ffmult, conv_expansion_factor=self.exp_fac, conv_kernel_size=self.kernel_size),
                                             self.n_encoders)

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        feats = x
        # post-processing on front-end features
        # x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        # x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = x.squeeze(dim=1)
        # Conformer encoder
        # x = self.encoder(x)
        for layer in self.encoder_conformer:
            x = layer(x)
        # post-processing on front-end features
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.encoder(x)

        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        if (self.is_train):
            return output, feats, last_hidden
        return output
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


class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ---------------------------------------- Res2Net ---------------------------------------- #


class Res2Net(nn.Module):
    def __init__(self, features_size, stride_=1, scale=4, padding_=1, groups_=1, reduction=16):
        super(Res2Net, self).__init__()
        # erro for wrong input
        if scale < 2 or features_size % scale:
            print('Error:illegal input for scale or feature size')

        self.divided_features = int(features_size / scale)
        self.conv1 = nn.Conv2d(features_size, features_size,
                               kernel_size=1, stride=stride_, padding=0, groups=groups_)
        self.conv2 = nn.Conv2d(self.divided_features, self.divided_features,
                               kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
        self.convs = nn.ModuleList()
        self.se = SEblock(features_size, reduction)
        for i in range(scale - 2):

            self.convs.append(
                nn.Conv2d(self.divided_features, self.divided_features,
                          kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
            )

    def forward(self, x):
        features_in = x
        conv1_out = self.conv1(features_in)
        y1 = conv1_out[:, 0:self.divided_features, :, :]
        fea = self.conv2(
            conv1_out[:, self.divided_features:2*self.divided_features, :, :])
        features = fea
        for i, conv in enumerate(self.convs):
            pos = (i + 1)*self.divided_features
            divided_feature = conv1_out[:, pos:pos+self.divided_features, :, :]
            fea = conv(fea + divided_feature)
            features = torch.cat([features, fea], dim=1)

        out = torch.cat([y1, features], dim=1)
        conv1_out1 = self.conv1(out)
        se_out = self.se(conv1_out1)
        result = features_in + se_out
        return result

# ---------------------------------------- Res2Net ---------------------------------------- #


class W2V2_RES2NET_COAASIST(nn.Module):
    def __init__(self, args, device, ssl_cpkt_path='./pretrained/xlsr2_300m.pt', emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=31, n_encoders=4):
        super().__init__()
        self.dim_head = int(emb_size/heads)
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####

        self.ssl_model = SSLModel(device, ssl_cpkt_path, 1024).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        # Replace the original Residual_block with the one from the AASIST model
        # to ConformerBlock
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            # nn.Sequential(Residual_block(nb_filts=filts[4])),
            # nn.Sequential(Residual_block(nb_filts=filts[4])),
            # nn.Sequential(Residual_block(nb_filts=filts[4]))
            nn.Sequential(Res2Net(64, 1, 4, 1, 1, 1)),
            nn.Sequential(Res2Net(64, 1, 4, 1, 1, 1)),
            nn.Sequential(Res2Net(64, 1, 4, 1, 1, 1)),
            nn.Sequential(Res2Net(64, 1, 4, 1, 1, 1)),
            nn.Sequential(Res2Net(64, 1, 4, 1, 1, 1))

        )

        self.encoder_conformer = _get_clones(ConformerBlock(dim=emb_size, dim_head=self.dim_head, heads=heads,
                                                            ff_mult=ffmult, conv_expansion_factor=exp_fac, conv_kernel_size=kernel_size),
                                             n_encoders)

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)

        # post-processing on front-end features
        # x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        # x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = x.squeeze(dim=1)
        # Conformer encoder
        # x = self.encoder(x)
        for layer in self.encoder_conformer:
            x = layer(x)
        # post-processing on front-end features
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))

        # RawNet2 encoder
        x = self.encoder(x)

        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output


