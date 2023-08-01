import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_ood.utils import OODMetrics, ToUnknown

import pandas as pd
import numpy as np
import os
from datautils.vocv4 import genList, Dataset_for, Dataset_for_eval
from model.wav2vec2_resnet import Model as wav2vec2_resnet
from model.wav2vec2_aasist import Model as wav2vec2_aasist
import yaml
from evaluate_metrics import compute_eer
from maxsoftmax import MaxSoftmax

from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc,
)




torch.manual_seed(1234)
device = "cuda:2"
DATABASE_DIR = "/dataa/Dataset/cnsl_real_fake_audio/"
# DATABASE_DIR = "xinwang_vocoders/data/voc.v4/"

# load config
config = yaml.load(open("configs/wav2vec2_aasist.yaml", "r"), Loader=yaml.FullLoader)
# load model
model = globals()[config['model']['name']](config['model'], device, emb=False)
# load state dict
model.load_state_dict(torch.load('out/model_weighted_CCE_60_8_1e-06_wav2vec2_aasist_jul6_balanceweight/epoch_59.pth',map_location=device))
model = model.to(device)

# Load test data
d_label_dev, file_dev = genList(dir_meta = os.path.join(DATABASE_DIR,'protocol_test.txt'),is_train=False,is_eval=False, is_dev=True)
    
print('no. of validation trials',len(file_dev))
    
dev_set = Dataset_for(config,list_IDs = file_dev,
	labels = d_label_dev,
	base_dir = os.path.join(DATABASE_DIR),algo=1)
test_loader = DataLoader(dev_set, batch_size=8,num_workers=8, shuffle=False)
# del dev_set,d_label_dev


detector = MaxSoftmax(model)
metrics = OODMetrics()
preds = []
for x, y in test_loader:
    score, pred = detector(x.to(device))
    metrics.update(score, y)
    # print(pred)
    preds.extend(pred.detach().cpu().tolist())

print(metrics.compute())
testdf = pd.read_csv(os.path.join(DATABASE_DIR, "protocol_test.txt"), sep=" ", header=None)
testdf.columns = ["path", "subset","label"]
testdf['score'] = metrics.buffer.get('scores')
testdf['pred'] = preds

scores = metrics.buffer.get('scores')
labels = metrics.buffer.get('labels')

known_scores = scores[labels == 0]
unknown_scores = scores[labels == 1]

eer, th = compute_eer(unknown_scores.cpu().numpy(), known_scores.cpu().numpy())
print(eer, th)
testdf.to_csv("testdf_aasist_jul6_balanceweight_59_wavefake.txt", sep=" ", index=False)