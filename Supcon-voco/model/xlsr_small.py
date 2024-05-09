import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
class SSLModel(nn.Module):
    def __init__(self, device='cpu', num_layers=24, order='first', custom_order=None):
        '''
        num_layers:
        '''
        super(SSLModel, self).__init__()
        self.num_layers = num_layers
        self.order = order
        self.custom_order = custom_order
        if self.num_layers < 1 or self.num_layers > 24:
            raise ValueError(
                "Number of layers must be at least 1 and at most 24.")
        
        cp_path = os.path.join(BASE_DIR,'pretrained/xlsr2_300m.pt')
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        # number of parameters
        print("Before cut: ", sum(p.numel() for p in model[0].parameters()))
        self.model = model[0]
        self.model = self.model.to(device)
        self.out_dim = 1024

        if self.order == 'last':
            # Get the last n layers
            self.model.encoder.layers = self.model.encoder.layers[-self.num_layers:]
        elif self.order == 'first':
            # Get the first n layers
            self.model.encoder.layers = self.model.encoder.layers[:self.num_layers]
        # elif self.order == 'middle':
        #     indices = middle_indices(24, self.num_layers)
            # self.model.encoder.layers = nn.ModuleList([
            #     self.model.encoder.layers[i] for i in indices])
        else:
            if self.custom_order is None:
                raise ValueError(
                    "Custom order must be provided as a list of integers (0-23).")

            # Check if the custom order is valid
            if type(self.custom_order) != list:
                raise ValueError("Custom order must be a list of integers.")

            # if len(self.custom_order) != self.num_layers:
            #     raise ValueError(
            #         "Length of custom order must be less than or equal to the number of layers.")
            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in self.custom_order])
        
        print("After cut: ", sum(p.numel() for p in self.model.parameters()))

    def extract_feat(self, input_data, is_train=True):
        
        # put the model to GPU if it not there
        # if next(self.model.parameters()).device != input_data.device \
        #    or next(self.model.parameters()).dtype != input_data.dtype:
        #     self.model.to(input_data.device, dtype=input_data.dtype)
        #     self.model.train()
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
            
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        # print(emb.shape)
        return emb
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLModel(device, num_layers=7, order='first')
    input_data = torch.randn(1, 16000).to(device)
    emb = model.extract_feat(input_data)
    # print(dir(emb))
    print(emb.shape)