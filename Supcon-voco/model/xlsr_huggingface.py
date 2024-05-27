import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
from transformers import AutoConfig, Wav2Vec2Model

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
class SSLModel(nn.Module):
    def __init__(self,device='cpu', num_layers=None, order='first', custom_order=None):
        super(SSLModel, self).__init__()
        self.is_train = True
        self.out_dim = 1024
        
        self.num_layers = num_layers
        self.order = order
        self.custom_order = custom_order
        # Speech pre-trained model
        self.config = AutoConfig.from_pretrained(
            'facebook/wav2vec2-xls-r-300m', 
            finetuning_task="audio-classification",
            revision="main",
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers
        
        self.model = Wav2Vec2Model.from_pretrained(
            'facebook/wav2vec2-xls-r-300m',
            from_tf=bool(".ckpt" in 'facebook/wav2vec2-xls-r-300m'),
            config=self.config,
            revision="main",
            ignore_mismatched_sizes=False,
        ).to(device)
        
        
        print("After cut: ", sum(p.numel() for p in self.model.parameters()))
        
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # self.model = nn.parallel.DistributedDataParallel(
        #     model, device_ids=[device], find_unused_parameters=True
        # )
        
    def forward(self, x):
        """
        x: (batch, length)
        """
        if(self.is_train):
            self.model.train()
            for param in self.model.parameters():
                param.require_grad = True
            x = self.model(x).last_hidden_state 
            
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.require_grad = False
            with torch.no_grad():
                x = self.model(x).last_hidden_state
        
        return x
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLModel(device, num_layers=7, order='first')
    input_data = torch.randn(1, 16000).to(device)
    emb = model(input_data)
    # print(len(emb))
    # print(dir(emb))
    print(emb.shape)