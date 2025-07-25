import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from core_scripts.startup_config import set_random_seed

# from model.wav2vec2_resnet import Model as wav2vec2_resnet
# from model.wav2vec2_resnet_contraall import Model as wav2vec2_resnet_contraall
# from model.wav2vec2_aasist import Model as wav2vec2_aasist
# from model.wav2vec2_linear import Model as wav2vec2_linear
# from model.wav2vec2_linear_nll_multi import Model as wav2vec2_linear_nll
# from model.wav2vec2_resnet_nll import Model as wav2vec2_resnet_nll
# from model.wav2vec2_linear_nll_4 import Model as wav2vec2_linear_nll_4
# from model.wav2vec2_mixup_linear import Model as wav2vec2_mixup_linear
# from model.wav2vec2_mixup4_linear import Model as wav2vec2_mixup4_linear
# from model.wav2vec2_mixup5_linear import Model as wav2vec2_mixup5_linear
# from model.wav2vec2_mixup6_linear import Model as wav2vec2_mixup6_linear
# from model.wav2vec2_vib import Model as wav2vec2_vib
# from model.wav2vec2_linear_nll_gelu import Model as wav2vec2_linear_nll_gelu
# from model.wav2vec2_vib_gelu import Model as wav2vec2_vib_gelu
# from model.wav2vec2_vib_gelu_multi import Model as wav2vec2_vib_gelu_multi
# from model.wav2vec2_coaasist import W2V2_COAASIST as wav2vec2_coaasist
# try:
#     from model.wav2vec2_btse import wav2vec2_btse
# except:
#     print("No dependency for wav2vec2_btse. Please switch to conda env bio")
from model.wav2vec2_btse import wav2vec2_btse
import importlib
import time

from model.loss_metrics import loss_custom
from tensorboardX import SummaryWriter

__author__ = "PhucDT"
__reference__ = "Hemlata Tak"

class EarlyStop:
    def __init__(self, patience=5, delta=0, init_best=60, save_dir=''):
        self.patience = patience
        self.delta = delta
        self.best_score = init_best
        self.init_best = init_best
        self.counter = 0
        self.early_stop = False
        self.save_dir = save_dir

    def __call__(self, score, model, epoch):
        
        if self.init_best < 0:
            # save every epoch
            torch.save(model.state_dict(), os.path.join(
                self.save_dir, 'epoch_{}.pth'.format(epoch)))
            
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("Best epoch: {}".format(epoch))
            self.best_score = score
            self.counter = 0
            # save model here
            if self.init_best > 0:
                torch.save(model.state_dict(), os.path.join(
                    self.save_dir, 'epoch_{}.pth'.format(epoch)))

        
def train_epoch(train_loader, model, lr, optimizer, device, config):
    model.train()
    running_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    train_loss_detail = {}
    for info, batch_x, batch_y in tqdm(train_loader, ncols=100):
        train_loss = 0.0
        # print("training on anchor: ", info)
        # check number of dimension of batch_x
        # print('batch_y', batch_y)
        # print('batch_x', batch_x.shape)
        batch_x = batch_x.to(device)
        if len(batch_x.shape) == 3:
            batch_x = batch_x.squeeze(0).transpose(0,1)
        elif len(batch_x.shape) == 4:
            batch_x = batch_x.squeeze(0)

        # print("batch_y", batch_y)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # print('batch_y before', batch_y)
        

        # for mixup
        if 'mixup' in config['model']['name']:
            batch_out, batch_feat, batch_emb, batch_y = model(batch_x, batch_y)
            # print("Batch_y: ", batch_y)
        else:
            batch_out, batch_feat, batch_emb = model(batch_x)
        # losses = loss_custom(batch_out, batch_feat, batch_emb, batch_y, config)
        losses = model.loss(batch_out, batch_feat, batch_emb, batch_y, config, info)
        for key, value in losses.items():
            train_loss += value
            train_loss_detail[key] = train_loss_detail.get(key, 0) + value.item()
            
        running_loss+=train_loss.item()
        # print('batch_y', batch_y.shape)
        
        if batch_out.dim() == 3:
            # duplicate batch_y
            batch_y = batch_y.repeat(batch_out.shape[1])
            batch_out = batch_out.view(batch_out.shape[0]*batch_out.shape[1], -1)
            
        _, batch_pred = batch_out.max(dim=-1)
        # batch_y = batch_y.view(-1)
        # print('batch_y', batch_y.shape)
        num_total +=batch_y.shape[0]
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        # print('num_correct', (batch_pred == batch_y).sum(dim=0).item())
        # print('batch_y', batch_y)
        # print('batch_out', batch_out)
        # print('batch_pred', batch_pred)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy, train_loss_detail

def evaluate_accuracy(dev_loader, model, device, config):
    val_loss = 0.0
    val_loss_detail = {}
    num_total = 0.0
    num_correct = 0.0
    model.eval()
    with torch.no_grad():
        for info, batch_x, batch_y in tqdm(dev_loader, ncols=100):
            loss_value = 0.0
            # print("Validating on anchor: ", info)

            batch_x = batch_x.to(device)
            if len(batch_x.shape) == 3:
                batch_x = batch_x.squeeze(0).transpose(0,1)
            elif len(batch_x.shape) == 4:
                batch_x = batch_x.squeeze(0)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            if 'mixup' in config['model']['name']:
                batch_out, batch_feat, batch_emb, batch_y = model(batch_x, batch_y)
            else:
                batch_out, batch_feat, batch_emb = model(batch_x)
            # losses = loss_custom(batch_out, batch_feat, batch_emb, batch_y, config)
            losses = model.loss(batch_out, batch_feat, batch_emb, batch_y, config, info)
            for key, value in losses.items():
                loss_value += value
                val_loss_detail[key] = val_loss_detail.get(key, 0) + value.item()
            
            if batch_out.dim() == 3:
                 # duplicate batch_y
                batch_y = batch_y.repeat(batch_out.shape[1])
                batch_out = batch_out.view(batch_out.shape[0]*batch_out.shape[1], -1)
               
            
            num_total +=batch_y.shape[0]
            val_loss+=loss_value.item()
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

    # val_loss /= num_total
    val_accuracy = (num_correct/num_total)*100

    return val_loss, val_accuracy, val_loss_detail

def produce_emb_file(dataset, model, device, save_path, batch_size=10):
    data_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    model.is_train = True

    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader, ncols=100):
            fname_list = []
            score_list = []
            pred_list = []
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)

            batch_out, feats, batch_emb = model(batch_x)
            score_list.extend(batch_out.data.cpu().numpy().tolist())
            # add outputs
            fname_list.extend(utt_id)
            # latent is the 4th element in feats tuple
            latent = feats[3]
            # predict label
            _,batch_pred = batch_out.max(dim=1)
            # print(batch_out)
            pred_list.extend(batch_pred.tolist())

            # save_path now must be a directory
            # make dir if not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Then each emb should be save in a file with name is utt_id
            for f, emb in zip(fname_list,latent):
                # normalize filename
                # f = f.split('/')[-1].split('.')[0] # utt id only
                save_path_utt = os.path.join(save_path, f)
                # check the folder exist
                folder = os.path.dirname(save_path_utt)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(save_path_utt, emb.data.cpu().numpy())

            # score file save into a single file
            with open(os.path.join(save_path, "scores.txt"), 'a+') as fh:
                for f, pr, cm in zip(fname_list,pred_list, score_list):
                    fh.write('{} {} {}\n'.format(f, pr, cm))
            fh.close()
    print('Scores saved to {}'.format(save_path))
    
def extract_features(data_loader, model, device, save_path, batch_size=10):
    # data_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    model.is_train = True

    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for utt_id, batch_x, _ in tqdm(data_loader, ncols=100):
            fname_list = []
            score_list = []
            pred_list = []
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)

            batch_out, _, batch_emb = model(batch_x)
            score_list.extend(batch_out.data.cpu().numpy().tolist())
            # add outputs
            fname_list.extend(utt_id)
            
            # predict label
            _,batch_pred = batch_out.max(dim=1)
            # print(batch_out)
            pred_list.extend(batch_pred.tolist())

            # save_path now must be a directory
            # make dir if not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Then each emb should be save in a file with name is utt_id
            for f, emb in zip(fname_list,batch_emb):
                # normalize filename
                # f = f.split('/')[-1].split('.')[0] # utt id only
                save_path_utt = os.path.join(save_path, f)
                # check the folder exist
                folder = os.path.dirname(save_path_utt)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(save_path_utt, emb.data.cpu().numpy())

    #         # score file save into a single file
    #         with open(os.path.join(save_path, "scores.txt"), 'a+') as fh:
    #             for f, pr, cm in zip(fname_list,pred_list, score_list):
    #                 fh.write('{} {} {}\n'.format(f, pr, cm))
    #         fh.close()
    # print('Scores saved to {}'.format(save_path))

def produce_evaluation_file(dataset, model, device, save_path, batch_size=10):
    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    model.eval()
    model.is_train = False
    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader, ncols=100):
            batch_x = batch_x.to(device)
            
            batch_out = model(batch_x)
            # batch_score = batch_out[:, 1].cpu().numpy().ravel()

            # add outputs
            fname_list = list(utt_id)
            # score_list = batch_out.data.cpu().numpy().tolist()
            pred_list = batch_out.max(dim=1)[1].tolist()
            pred_score = batch_out.max(dim=1)[0].tolist()

            with open(save_path, 'a+') as fh:
                for f, pred, score in zip(fname_list, pred_list, pred_score):
                    fh.write('{} {} {}\n'.format(f, pred, score))

    print('Scores saved to {}'.format(save_path))

def produce_prediction_file(dataset, model, device, save_path, batch_size=10):
    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    # set to inference mode
    model.is_train = False

    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x, utt_id in data_loader:
            fname_list = []
            score_list = []
            pred_list = []
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)

            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]
                        ).data.cpu().numpy().ravel()
            _,batch_pred = batch_out.max(dim=1)
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            pred_list.extend(batch_pred.tolist())

            with open(save_path, 'a+') as fh:
                for f, cm, pred in zip(fname_list,score_list, pred_list):
                    fh.write('{} {} {}\n'.format(f, cm, pred))
            fh.close()
    print('Scores saved to {}'.format(save_path))

def load_state_dict_with_mismatch_handling(model, state_dict):
    model_dict = model.state_dict()

    # Iterate over the state dict
    for name, param in state_dict.items():
        if name in model_dict:
            # Check for size mismatch
            if model_dict[name].size() == param.size():
                model_dict[name].copy_(param)
            else:
                print(f"Size mismatch for {name}: expected {model_dict[name].size()}, got {param.size()}. Initializing randomly.")
                if isinstance(model_dict[name], nn.Parameter):
                    torch.nn.init.normal_(model_dict[name])

    model.load_state_dict(model_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/', help='eval set')
    '''
    % database_path/
    %   | - protocol.txt
    %   | - audio_path
    '''
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=0.0000001)
    parser.add_argument('--max_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--padding_type', type=str, default='zero',
                        help='zero or repeat')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='training dataloader or dev dataloader')
    # model
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')

    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='get the predicted label instead of score')
    parser.add_argument('--emb', action='store_true', default=False,
                        help='produce embeddings')
    parser.add_argument('--extract_features', default=None,
                        help='Extract features from the model')
    parser.add_argument('--dev_diff', action='store_true', default=False,
                        help='If True, use the different dev loader. Default is same to train loader')

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5,
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5,
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20,
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000,
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100,
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000,
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10,
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100,
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0,
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0,
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5,
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10,
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2,
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40,
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')

    ##===================================================Rawboost data augmentation ======================================================================#


    if not os.path.exists('out'):
        os.mkdir('out')
    args = parser.parse_args()

    # set random seed
    set_random_seed(args.seed)
    # torch.manual_seed(args.seed)

    # #define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(
        args.loss, args.num_epochs, args.batch_size, args.min_lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('out', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    #GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Number of GPUs available: ", torch.cuda.device_count())

    # load config file
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    # dynamic load datautils based on name in config file
    genList = importlib.import_module('datautils.'+config['data']['name']).genList
    Dataset_for = importlib.import_module('datautils.'+config['data']['name']).Dataset_for
    Dataset_for_eval = importlib.import_module('datautils.'+config['data']['name']).Dataset_for_eval

    if args.dev_diff:
        print("Dev_DIFF mode enabled! Developing is load from normal dataloader")
        Dataset_for_dev = Dataset_for_eval

    # dynamic load model based on name in config file
    modelClass = importlib.import_module('model.'+config['model']['name']).Model
    model = modelClass(config['model'], device)
    # model = globals()[config['model']['name']](config['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    print('nb_params:',nb_params)

    
    # load state dict
    if args.model_path:
        # fix state dict missing and unexpected keys
        pretrained_dict = torch.load(args.model_path)
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        pretrained_dict = {key.replace("_orig_mod.", ""): value for key, value in pretrained_dict.items()}
        # model.load_state_dict(pretrained_dict, strict=False)
        load_state_dict_with_mismatch_handling(model, pretrained_dict)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        print('Model loaded')
    else:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        print('Model initialized')
    #evaluation
    
    if args.eval:
        _,file_eval = genList(dir_meta =  os.path.join(args.database_path,'protocol.txt'),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_for_eval(list_IDs = file_eval, base_dir = os.path.join(args.database_path+'/'), max_len=config['data']['kwargs']['trim_length'], padding_type=args.padding_type, **config['data']['kwargs'])
        if (args.predict):
            produce_prediction_file(eval_set, model, device, args.eval_output, batch_size=args.batch_size)
        elif (args.emb):
            produce_emb_file(eval_set, model, device, args.eval_output, batch_size=args.batch_size)
        else:
            produce_evaluation_file(eval_set, model, device, args.eval_output, batch_size=args.batch_size)
        sys.exit(0)


    # define train dataloader
    # d_label_trn,file_train = genList(dir_meta = args.database_path),is_train=True,is_eval=False,is_dev=False)
    d_label_trn,file_train = genList( dir_meta = os.path.join(args.database_path,'protocol.txt'),is_train=True,is_eval=False,is_dev=False)
    # check config['data']['portion'] is defined or not
    if 'portion' in config['data']:
        idx = range(len(file_train))
        idx = np.random.choice(idx, int(len(file_train)*config['data']['portion']), replace=False)
        file_train = [file_train[i] for i in idx]
        if len(d_label_trn)>0: # the case of train without label
            d_label_trn = {k: d_label_trn[k] for k in file_train}
    print('no. of training trials',len(file_train))
    is_repeat_pad = True if args.padding_type=='repeat' else False
    train_set=Dataset_for(args, list_IDs = file_train, labels = d_label_trn,
        base_dir = args.database_path+'/',algo=args.algo, repeat_pad=is_repeat_pad, is_train=True, **config['data']['kwargs'])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)

    del train_set,d_label_trn


    # # define validation dataloader

    # d_label_dev,file_dev = genList(dir_meta = os.path.join(args.database_path,'protocol.txt'),is_train=False,is_eval=False, is_dev=True)

    # print('no. of validation trials',len(file_dev))
    # args.is_train = False
    # dev_set = Dataset_for(args,list_IDs = file_dev, labels = d_label_dev,
	# 	base_dir = args.database_path+'/',algo=args.algo, repeat_pad=is_repeat_pad, is_train=False, **config['data']['kwargs'])

    # dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    # del dev_set,d_label_dev
    # define validation dataloader

    if args.dev_diff:
        # change the genList of Dev set to normal dataloader
        genList = importlib.import_module('datautils.normal').genList

    d_label_dev,file_dev = genList(dir_meta =os.path.join(args.database_path,'protocol.txt'),is_train=False,is_eval=False, is_dev=True)
    if 'portion' in config['data']:
        idx = range(len(file_dev))
        idx = np.random.choice(idx, int(len(file_dev)*config['data']['portion']), replace=False)
        file_dev = [file_dev[i] for i in idx]
        if len(d_label_dev)>0: # the case of train without label
            d_label_dev = {k: d_label_dev[k] for k in file_dev}
    print('no. of validation trials',len(file_dev))
    args.is_train = False
    if args.dev_diff:
        dev_set=Dataset_for_dev(args,list_IDs = file_dev, labels = d_label_dev,
            base_dir = args.database_path+'/',algo=args.algo, repeat_pad=is_repeat_pad, is_train=False, **config['data']['kwargs'])
        dev_loader = DataLoader(dev_set, batch_size=args.batch_size*32,num_workers=8, shuffle=False)
    else:
        dev_set = Dataset_for(args,list_IDs = file_dev, labels = d_label_dev,
            base_dir = args.database_path+'/',algo=args.algo, repeat_pad=is_repeat_pad, is_train=False, **config['data']['kwargs'])
        dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)


    del dev_set,d_label_dev
    
    if args.extract_features:
        # assume that loader is normal loader
        # extract train features
        print("Extracting train features")
        extract_features(train_loader, model, device, args.extract_features, batch_size=args.batch_size)
        # extract dev features
        print("Extracting dev features")
        extract_features(dev_loader, model, device, args.extract_features, batch_size=args.batch_size)
        sys.exit(0)
    
    #set Adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.min_lr,weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr, cycle_momentum=False)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr,step_size_up=2,mode="exp_range",gamma=0.85, cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1, eta_min=args.min_lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=20, steps_per_epoch = len(train_loader))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.5)


    # Training and validation
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    early_stopping = EarlyStop(patience=50, delta=0.001, init_best=-1, save_dir=model_save_path)
    start_train_time = time.time()
    for epoch in range(args.start_epoch,args.start_epoch + num_epochs, 1):
        print('Epoch {}/{}. Current LR: {}'.format(epoch, num_epochs - 1, optimizer.param_groups[0]['lr']))

        running_loss, train_accuracy, train_loss_detail = train_epoch(train_loader, model, args.min_lr, optimizer, device, config)
        val_loss, val_accuracy, val_loss_detail = evaluate_accuracy(dev_loader, model, device, config)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('val_accuracy', val_accuracy, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        for loss_name, loss in train_loss_detail.items():
            writer.add_scalar('train_{}'.format(loss_name), loss, epoch)
        for loss_name, loss in val_loss_detail.items():
            writer.add_scalar('val_{}'.format(loss_name), loss, epoch)
        print('\n{} - {} - {}\nTraining ACC:{} - Validation ACC: {}'.format(epoch,running_loss,val_loss, train_accuracy, val_accuracy))
        scheduler.step()
        # check early stopping
        early_stopping(val_accuracy, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping activated.")
            break

    print("Total training time: {}s".format(time.time() - start_train_time))
