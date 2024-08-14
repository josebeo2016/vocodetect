import torch
from torch import nn
import torch.nn.functional as F
import os
try:
    from .model import WavLM, WavLMConfig
except:
    from model import WavLM, WavLMConfig
    
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
class WavLMFe(nn.Module):
    def __init__(self, **kwargs):
        super(WavLMFe, self).__init__()
        # load the pre-trained checkpoints
        checkpoint_path = kwargs.get(
            'checkpoint_path', 'model/pretrained/WavLM-Large.pt')
        # 'last', 'all', 'weighted_sum'
        checkpoint = torch.load(os.path.join(BASE_DIR,checkpoint_path))
        cfg = WavLMConfig(checkpoint['cfg'])

        self.extract_mode = kwargs.get('extract_mode', 'last')
        self.cfg = cfg
        self.num_layers = cfg.encoder_layers + 1
        self.model = WavLM(cfg)
        self.model.load_state_dict(checkpoint['model'])
        print('Loaded pre-trained WavLM from', checkpoint_path)
        self.out_dim = cfg.encoder_embed_dim
        self.is_train = True
        if self.extract_mode == 'weighted_sum':
            # initialize the weights for weighted sum
            print('Initializing weights for weighted sum')
            self.weights = nn.Parameter(torch.ones(
                self.num_layers), requires_grad=True)
        elif self.extract_mode == 'attentive_merging':
            print('Initializing weights for attentive merging')
            # Initialize components for attentive merging
            self.W_sq = nn.Parameter(torch.randn(cfg.encoder_embed_dim, 1))
            self.W_ex1 = nn.Parameter(torch.randn(self.num_layers, self.num_layers // 2))
            self.W_ex2 = nn.Parameter(torch.randn(self.num_layers // 2, self.num_layers))
            self.W_L1 = nn.Parameter(torch.randn(self.num_layers * cfg.encoder_embed_dim, (self.num_layers * cfg.encoder_embed_dim) // 4))
            self.W_L2 = nn.Parameter(torch.randn((self.num_layers * cfg.encoder_embed_dim) // 4, (self.num_layers * cfg.encoder_embed_dim) // 4))
            self.W_L3 = nn.Parameter(torch.randn((self.num_layers * cfg.encoder_embed_dim) // 4, cfg.encoder_embed_dim))


    def forward(self, wav_input_16khz):
        if self.cfg.normalize:
            wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
        if self.extract_mode == 'weighted_sum':
            return self.forward_with_intermediate(wav_input_16khz)
        elif self.extract_mode == 'attentive_merging':
            return self.forward_with_attentive_merging(wav_input_16khz)
        else:
            return self.model.extract_features(wav_input_16khz)[0]

    def forward_with_intermediate(self, wav_input_16khz):
        if self.is_train:
            self.model.train()
        else:
            self.model.eval()
        rep, layer_results = self.model.extract_features(
            wav_input_16khz, output_layer=self.model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

        weighted_layer_reps = [x * w for x, w in zip(layer_reps, self.weights)]
        return torch.sum(torch.stack(weighted_layer_reps), dim=0)

    def forward_with_attentive_merging(self, wav_input_16khz):
        rep, layer_results = self.model.extract_features(
            wav_input_16khz, output_layer=self.model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        #print("🐍 File: WavLM/fe.py | Line: 62 | forward_with_attentive_merging ~ layer_reps", len(layer_reps))
        
        # Stack all hidden embeddings
        X = torch.stack(layer_reps, dim=-1)  # Shape: (B, T, H, L)
        #print("🐍 File: WavLM/fe.py | Line: 65 | forward_with_attentive_merging ~ X", X.shape)

        # Step 1: Average across the time dimension
        X_avg = torch.mean(X, dim=1)  # Shape: (B, H, L)
        #print("🐍 File: WavLM/fe.py | Line: 68 | forward_with_attentive_merging ~ X_avg", X_avg.shape)

        # Step 2: Fully connected layer with SWISH activation
        X_avg_transposed = X_avg.transpose(1, 2)  # Shape: (B, L, H)
        x_sq = torch.matmul(X_avg_transposed, self.W_sq)  # (B, L, H) * (H, 1) Shape: (B, L, 1)
        x_sq = torch.nn.functional.silu(x_sq).squeeze(-1)  # Shape: (B, L)
        #print("🐍 File: WavLM/fe.py | Line: 76 | forward_with_attentive_merging ~ x_sq",x_sq.shape)


        # Step 3: Obtain attentive weights
        x_attW = torch.sigmoid(torch.matmul(x_sq, self.W_ex1))  # Shape: (B, H, L//2)
        x_attW = torch.matmul(x_attW, self.W_ex2)  # Shape: (B, H, L)

        # Step 4: Apply attentive weights
        x_attW = x_attW.unsqueeze(1).unsqueeze(1)  # Shape: (B, 1, 1, L)
        X_att = X * x_attW  # Shape: (B, T, H, L)

        # Step 5: Concatenate all embeddings and merge using 3-layer linear projection network
        X_att = X_att.view(X_att.size(0), X_att.size(1), -1)  # Shape: (B, T, H * L)
        #print("🐍 File: WavLM/fe.py | Line: 89 | forward_with_attentive_merging ~ X_att",X_att.shape)
        X_attM = F.linear(X_att, self.W_L1.transpose(0, 1))
        X_attM = F.linear(F.silu(X_attM), self.W_L2)
        X_attM = F.linear(F.silu(X_attM), self.W_L3.transpose(0, 1))  # Shape: (B, T, H)

        return X_attM


if __name__ == '__main__':
    
    fe = WavLMFe(extract_mode='weighted_sum', checkpoint_path='/dataa/phucdt/vocodetect/Supcon-voco/model/pretrained/WavLM-Large.pt')
  
    input_tensor = torch.randn(3, 16000)
    out = fe(input_tensor)
    print(out.shape)
#     # loss
#     ce_loss = nn.CrossEntropyLoss()
#     torch.autograd.set_detect_anomaly(True) # for debugging
#     # forward
#     output = fe(input_tensor) # torch.Size([1, 49, 768])
#     print(output.shape)
#     # convert output to 2D tensor
#     output = output.view(output.size(0), -1)

#     # full connected layer
#     fc = nn.Linear(49*1024, 2)

#     final_output = fc(output)

#     print(final_output.shape)
    
#     loss = ce_loss(final_output, torch.tensor([1]))

#     print(loss)

#     # loss 

#     loss.backward()