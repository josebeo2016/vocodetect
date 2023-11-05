import os
import tqdm
from pathlib import Path
import pandas as pd
import argparse
import random
import glob 

def recursive_list_files(path, file_extension=".wav"):
    """Recursively lists all files in a directory and its subdirectories"""
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                files.append(os.path.join(dirpath, filename))
    return files
def genpath(val, path, ext=".flac"):
    return os.path.join(path, val+ext)

def subset(row):
    if "_T_" in row['path']:
        return 'train'
    else:
        return 'dev'

def jul6():
    rate = 0.5
    
    # ==================== #
    # make proto for yttts
    # ==================== #
    yttts_files = recursive_list_files("jul6/yttts", file_extension=".wav")
    protocol_yttts = pd.DataFrame(yttts_files, columns=['path'])
    protocol_yttts['label'] = "bonafide"
    protocol_yttts['path'] = protocol_yttts['path'].apply(lambda x: x.replace("jul6/", ""))
    protocol_yttts = protocol_yttts[:7500]
    print(protocol_yttts.head)
    
    # ==================== #
    # In the wild
    # ==================== #
    in_the_wild_bona = recursive_list_files("jul6/in_the_wild_bona", file_extension=".wav")
    protocol_in_the_wild_bona = pd.DataFrame(in_the_wild_bona, columns=['path'])
    protocol_in_the_wild_bona['path'] = protocol_in_the_wild_bona['path'].apply(lambda x: x.replace("jul6/", ""))
    protocol_in_the_wild_bona['label'] = "bonafide"
    protocol_in_the_wild_bona = protocol_in_the_wild_bona[:7500]
    print(protocol_in_the_wild_bona.head())
    
    # ==================== #
    # in the wild voco
    # ==================== #
    
    # ==================== #
    # xinwang vocv4
    # ==================== #
    vocv4_ori = pd.read_csv("/dataa/phucdt/vocodetect/xinwang_vocoders/data/voc.v4/ori_protocol.txt", sep=" ", header=None)
    protocol_vocv4 = pd.DataFrame(vocv4_ori[1].apply(lambda x: "vocv4/"+x+".wav").values, columns=['path'])
    protocol_vocv4['label'] = vocv4_ori[3].values
    print(protocol_vocv4.head())
    
    # ==================== #
    # LA2019 bonafide
    # ==================== #
    la2019_bona = recursive_list_files("jul6/la2019_bona", file_extension=".wav")
    protocol_la2019_bona = pd.DataFrame(la2019_bona, columns=['path'])
    protocol_la2019_bona['path'] = protocol_la2019_bona['path'].apply(lambda x: x.replace("jul6/", ""))
    protocol_la2019_bona['label'] = "bonafide"
    
    # ==================== #
    # Korean real
    # ==================== #
    
    
    # ==================== #
    # merge all proto
    # but keep label and path only
    # ==================== #
    protocol = pd.concat([protocol_yttts, protocol_in_the_wild_bona, protocol_vocv4, protocol_la2019_bona], ignore_index=True)
    print(protocol.head())
    
    # ==================== #
    # shuffle and random assign to train, dev
    # ==================== #
    protocol_bonafide = protocol[protocol.label == "bonafide"]
    protocol_spoof = protocol[protocol.label != "bonafide"]
    
    # # shuffle
    protocol_bonafide = protocol_bonafide.sample(frac=1, random_state=1234, ignore_index=True)
    print(protocol_bonafide.head())
    protocol_spoof = protocol_spoof.sample(frac=1, random_state=1234, ignore_index=True)
    print(protocol_spoof.head())
    
    
    # merge
    protocol = pd.concat([protocol_bonafide, protocol_spoof], ignore_index=True)
    
    protocol = protocol.sample(frac=1, random_state=1234, ignore_index=True)
    protocol['subset'] = "train"
    protocol["subset"][protocol.index >= int(len(protocol)*rate)] = "dev"
    
    print(protocol.value_counts("subset"))
    print(protocol.value_counts("label"))
    
    print("Breakdown:")
    print("DEV: ", protocol[protocol.subset == "dev"].value_counts("label"))
    print("TRAIN: ", protocol[protocol.subset == "train"].value_counts("label"))
    
    # print("Total Number of train: ", len(protocol[protocol.subset == "train"]))
    # print("Total spoof in train: ", len(protocol[(protocol.subset == "train") & (protocol.label == "spoof")]))
    # print("Total bonafide in train: ", len(protocol[(protocol.subset == "train") & (protocol.label == "bonafide")]))
    # print("Total Number of dev: ", len(protocol[protocol.subset == "dev"]))
    # print("Total spoof in dev: ", len(protocol[(protocol.subset == "dev") & (protocol.label == "spoof")]))
    # print("Total bonafide in dev: ", len(protocol[(protocol.subset == "dev") & (protocol.label == "bonafide")]))
    
    # ==================== #
    # save proto
    # ==================== #
    protocol[['path', 'subset', 'label']].to_csv("jul6/protocol.txt", sep=' ', index=False, header=False)
    protocol[protocol.subset == 'dev'][['path', 'subset', 'label']].to_csv("jul6/protocol_test.txt", sep=' ', index=False, header=False)
    print("Done!")
    
def jul11():
    rate = 0.5
    # ==================== #
    # in_the_wild_bona
    # ==================== #
    in_the_wild_filelist = recursive_list_files("jul11/in_the_wild_bona", file_extension=".wav")
    protocol_in_the_wild_bona = pd.DataFrame(in_the_wild_filelist, columns=['path'])
    protocol_in_the_wild_bona['path'] = protocol_in_the_wild_bona['path'].apply(lambda x: x.replace("jul11/", ""))
    protocol_in_the_wild_bona['label'] = "bonafide"
    
    # split into train and dev
    protocol_in_the_wild_bona = protocol_in_the_wild_bona.sample(frac=1, random_state=1234, ignore_index=True)
    protocol_in_the_wild_bona['subset'] = "train"
    protocol_in_the_wild_bona["subset"][protocol_in_the_wild_bona.index >= int(len(protocol_in_the_wild_bona)*rate)] = "dev"
    
    # ==================== #
    # Duplicate in_the_wild_bona to make in_the_wild_hifigan
    # ==================== #
    protocol_in_the_wild_hifigan = protocol_in_the_wild_bona.copy()
    protocol_in_the_wild_hifigan['path'] = protocol_in_the_wild_hifigan['path'].apply(lambda x: x.replace("in_the_wild_bona", "in_the_wild_voco/hifigan"))
    protocol_in_the_wild_hifigan['label'] = "hifigan"
    
    # ==================== #
    # Duplicate in_the_wild_bona to make in_the_wild_hn_sinc_nsf
    # ==================== #
    protocol_in_the_wild_hn_sinc_nsf = protocol_in_the_wild_bona.copy()
    protocol_in_the_wild_hn_sinc_nsf['path'] = protocol_in_the_wild_hn_sinc_nsf['path'].apply(lambda x: x.replace("in_the_wild_bona", "in_the_wild_voco/hn_sinc_nsf"))
    protocol_in_the_wild_hn_sinc_nsf['label'] = "hn_sinc_nsf"
    
    # ==================== #
    # Duplicate in_the_wild_bona to make in_the_wild_hn_sinc_nsf_hifi
    # ==================== #
    protocol_in_the_wild_hn_sinc_nsf_hifi = protocol_in_the_wild_bona.copy()
    protocol_in_the_wild_hn_sinc_nsf_hifi['path'] = protocol_in_the_wild_hn_sinc_nsf_hifi['path'].apply(lambda x: x.replace("in_the_wild_bona", "in_the_wild_voco/hn_sinc_nsf_hifi"))
    protocol_in_the_wild_hn_sinc_nsf_hifi['label'] = "hn_sinc_nsf_hifi"
    
    # ==================== #
    # Duplicate in_the_wild_bona to make in_the_wild_waveglow
    # ==================== #
    protocol_in_the_wild_waveglow = protocol_in_the_wild_bona.copy()
    protocol_in_the_wild_waveglow['path'] = protocol_in_the_wild_waveglow['path'].apply(lambda x: x.replace("in_the_wild_bona", "in_the_wild_voco/waveglow"))
    protocol_in_the_wild_waveglow['label'] = "waveglow"
    
    # ==================== #
    # la2019 data
    # ==================== #
    la2019_voco = pd.read_csv("../xinwang_vocoders/data/voc.v4/ori_protocol.txt", sep=' ', header=None)
    protocol_la2019_voco = pd.DataFrame(la2019_voco[3].values, columns=['label'])
    protocol_la2019_voco['path'] = la2019_voco[1].apply(lambda x: "la2019_voco/"+x)
    protocol_la2019_voco['subset'] = protocol_la2019_voco.apply(subset, axis=1)
    print(protocol_la2019_voco.head)
    protocol_la2019_bona = protocol_la2019_voco[protocol_la2019_voco.label == "hifigan"].copy()
    protocol_la2019_bona['label'] = "bonafide"
    protocol_la2019_bona['path'] = protocol_la2019_bona['path'].apply(lambda x: x.replace("la2019_voco/hifigan_", "la2019_voco/"))
    
    print(protocol_la2019_bona.head)
    
    # merge all protocol
    protocol = pd.concat([protocol_in_the_wild_bona, protocol_in_the_wild_hifigan, protocol_in_the_wild_hn_sinc_nsf, protocol_in_the_wild_hn_sinc_nsf_hifi, protocol_in_the_wild_waveglow, protocol_la2019_bona, protocol_la2019_voco], ignore_index=True)
    print(protocol.head())
    print("Number of dev: ", len(protocol[protocol.subset == "dev"]))
    print("Number of train: ", len(protocol[protocol.subset == "train"]))
    print("Number of bonafide in dev: ", len(protocol[(protocol.subset == "dev") & (protocol.label == "bonafide")]))
    print("Number of bonafide in train: ", len(protocol[(protocol.subset == "train") & (protocol.label == "bonafide")]))
    
    # Save protocol
    protocol.to_csv("jul11/protocol.txt", sep=' ', index=False, header=False)
    
def fakeav():
    # ==================== #
    # FakeAV Celeb protocol
    # ==================== #
    
    real = recursive_list_files("fakeav/audio_real", file_extension=".wav")
    fake = recursive_list_files("fakeav/audio_fake", file_extension=".wav")
    
    protocol_fakeav = pd.DataFrame(real, columns=['path'])
    protocol_fakeav['label'] = "bonafide"
    
    protocol_fakeav = pd.concat([protocol_fakeav, pd.DataFrame(fake, columns=['path'])], ignore_index=True)
    protocol_fakeav['label'][protocol_fakeav.label.isna()] = "spoof"
    
    protocol_fakeav['subset'] = "test"
    print("Total Number of spoof: ", len(protocol_fakeav[protocol_fakeav.label == "spoof"]))
    protocol_fakeav.to_csv("protocol_fakeav.txt", columns=['path', 'subset', 'label'], sep=' ', index=False, header=False)

def in_the_wild():
    # ==================== #
    # in_the_wild protocol
    # ==================== #
    protocol_in_the_wild = pd.read_csv("in_the_wild.txt", sep=" ", header=None)
    protocol_in_the_wild['path'] = 'in_the_wild/' + protocol_in_the_wild[0]
    protocol_in_the_wild['label'] = protocol_in_the_wild[1]
    protocol_in_the_wild['subset'] = "test"
    
    protocol_in_the_wild.to_csv("protocol_in_the_wild.txt", columns=['path', 'subset', 'label'], sep=' ', index=False, header=False)

def asvspoof_2021_df():
    # ==================== #
    # DF protocol
    # ==================== #
    protocol_df = pd.read_csv("/datab/Dataset/ASVspoof/LA/ASVspoof2021_DF_eval/keys/CM/trial_metadata.txt", sep=" ", header=None)
    protocol_df.columns = ['spk','utt','a','b','c', 'label', 'd', 'subset']
    protocol_df['path'] = "flac/" + protocol_df['utt'] + ".flac"
    protocol_df.to_csv("asvspoof_2021_DF/protocol.txt", columns=['path', 'subset', 'label'], sep=' ', index=False, header=False)
    
def wavefake():
    # ==================== #
    # wavefake protocol
    # ==================== #
    wavefake_files = recursive_list_files("generated_audio/", file_extension=".wav")
    protocol_wavefake = pd.DataFrame(wavefake_files, columns=['path'])
    protocol_wavefake['label'] = "spoof"
    protocol_wavefake['subset'] = "test"
    
    # ==================== #
    # LJSpeech
    # ==================== #
    ljspeech_files = recursive_list_files("LJSpeech/", file_extension=".wav")
    protocol_ljspeech = pd.DataFrame(ljspeech_files, columns=['path'])
    protocol_ljspeech['label'] = "bonafide"
    protocol_ljspeech['subset'] = "test"
    
    # ==================== #
    # merge all protocol
    # ==================== #
    protocol = pd.concat([protocol_wavefake, protocol_ljspeech], ignore_index=True)
    # shuffle
    protocol = protocol.sample(frac=1, random_state=1234, ignore_index=True)
    # pick 500 spoof and 500 bonafide
    protocol = pd.concat([protocol[protocol.label == "spoof"][:500], protocol[protocol.label == "bonafide"][:500]], ignore_index=True)
    # save
    protocol.to_csv("protocol_wavefake.txt", columns=['path', 'subset', 'label'], sep=' ', index=False, header=False)


def supcon_cnsl_sep30():
    rate = 0.5
    dataset_dir = '/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_sep30'
    bonafide_dir = os.path.join(dataset_dir,'bonafide')
    eval_dir = os.path.join(dataset_dir,'eval')
    scp_dir = os.path.join(dataset_dir,'scp')
    vocoded_dir = os.path.join(dataset_dir,'vocoded')
    
    # ==================== #
    # bonafide from la2019 (train dev)
    # ==================== #
    la2019_bona_dir = '/dataa/phucdt/vocodetect/traindata/asvspoof_2019_supcon/bonafide'
    # list all file in la2019_bona_dir
    la2019_bona_files = glob.glob(la2019_bona_dir+"/*.wav")
    la2019_bona_files = [os.path.basename(file) for file in la2019_bona_files]
    # write list to tmp file
    with open(os.path.join(scp_dir, '/tmp/tmp.txt'), 'w') as f:
        for file in la2019_bona_files:
            f.write(file+'\n')
    # softlink all files from la2019_bona_dir to bonafide_dir
    os.system("rsync -a --files-from='/tmp/tmp.txt' --info=progress2 "+ la2019_bona_dir+ " " +bonafide_dir)
    

    # ==================== #
    # vocoded from la2019 (train dev)
    # ==================== #
    la2019_voco_dir = '/dataa/phucdt/vocodetect/traindata/asvspoof_2019_supcon/vocoded'
    # list all file in la2019_voco_dir
    la2019_voco_files = glob.glob(la2019_voco_dir+"/*.wav")
    la2019_voco_files = [os.path.basename(file) for file in la2019_voco_files]
    # write list to tmp file
    with open(os.path.join(scp_dir, '/tmp/tmp.txt'), 'w') as f:
        for file in la2019_voco_files:
            f.write(file+'\n')
    # softlink all files from la2019_voco_dir to vocoded_dir
    os.system("rsync -a --files-from='/tmp/tmp.txt' --info=progress2 "+la2019_voco_dir+" "+vocoded_dir)
    
        
    # ==================== #
    # bonafide from CNSL_intern
    # ==================== #
    cnsl_bona_dir = '/datab/Dataset/CNSL_intern/norm'
    # list all file in cnsl_bona_dir
    cnsl_bona_files = glob.glob(cnsl_bona_dir+"/*.wav")
    cnsl_bona_files = [os.path.basename(file) for file in cnsl_bona_files]
    # write list to tmp file
    with open(os.path.join(scp_dir, '/tmp/tmp.txt'), 'w') as f:
        for file in cnsl_bona_files:
            f.write(file+'\n')
    # softlink all files from cnsl_bona_dir to bonafide_dir
    os.system("rsync -a --files-from='/tmp/tmp.txt' --info=progress2 "+cnsl_bona_dir+" "+bonafide_dir)
    
    
    # ==================== #
    # vocoded from CNSL_intern
    # ==================== #
    cnsl_voco_dir = '/datab/Dataset/CNSL_intern/vocoded'
    # list all file in cnsl_voco_dir
    cnsl_voco_files = glob.glob(cnsl_voco_dir+"/*.wav")
    cnsl_voco_files = [os.path.basename(file) for file in cnsl_voco_files]
    # write list to tmp file
    with open(os.path.join(scp_dir, '/tmp/tmp.txt'), 'w') as f:
        for file in cnsl_voco_files:
            f.write(file+'\n')
    # softlink all files from cnsl_voco_dir to vocoded_dir
    os.system("rsync -a --files-from='/tmp/tmp.txt' --info=progress2 "+cnsl_voco_dir+" "+vocoded_dir)
    
    
    # ==================== #
    # make scp train_bonafide.lst
    # ==================== #
    
    # mix la2019_bona and cnsl_bona
    bonafide_files = la2019_bona_files + cnsl_bona_files
    # shuffle
    random.shuffle(bonafide_files)
    # split train and dev
    bonafide_train = bonafide_files[:int(len(bonafide_files)*rate)]
    bonafide_dev = bonafide_files[int(len(bonafide_files)*rate):]
    
    # write lst file
    with open(os.path.join(scp_dir, 'train_bonafide.lst'), 'w') as f:
        for file in bonafide_train:
            f.write(file+'\n')
    with open(os.path.join(scp_dir, 'dev_bonafide.lst'), 'w') as f:
        for file in bonafide_dev:
            f.write(file+'\n')
            
    

def supcon_cnsl_oct24():
    rate = 0.5
    scp_dir = '/dataa/phucdt/vocodetect/traindata/supcon_cnsl_oct24/scp/'
    bonafide_dir = '/dataa/phucdt/vocodetect/traindata/supcon_cnsl_oct24/bonafide/'
    # Bonafide and vocoded from the supcon_cnsl_sep30
    # this code is use to split bonafide to train and dev
    
    # list bonafide files in '/dataa/phucdt/vocodetect/traindata/supcon_cnsl_oct24/bonafide'
    bonafide_files = glob.glob('/dataa/phucdt/vocodetect/traindata/supcon_cnsl_oct24/bonafide/*.wav')
    # shuffle
    random.shuffle(bonafide_files)
    # split train and dev
    bonafide_train = bonafide_files[:int(len(bonafide_files)*rate)]
    bonafide_dev = bonafide_files[int(len(bonafide_files)*rate):]
    
    # write lst file
    with open(os.path.join(scp_dir, 'train_bonafide.lst'), 'w') as f:
        for file in bonafide_train:
            f.write(file.replace(bonafide_dir,"")+'\n')
    with open(os.path.join(scp_dir, 'dev_bonafide.lst'), 'w') as f:
        for file in bonafide_dev:
            f.write(file.replace(bonafide_dir,"")+'\n')
    

if __name__ == "__main__":
    # jul6()
    # jul11()
    # main_df()
    # main_tts()
    # fakeav()
    # in_the_wild()
    # wavefake()
    # asvspoof_2021_df()
    # supcon_cnsl_sep30()
    supcon_cnsl_oct24()