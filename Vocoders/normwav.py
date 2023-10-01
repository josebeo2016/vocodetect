import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from scipy.io.wavfile import read
from utils.logging import get_logger
import librosa
import soundfile as sf

logger = get_logger(__file__)

def read_wav_np(path):
    sr, wav = read(path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    wav = wav.astype(np.float32)
    return sr, wav

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav_path', type=str, required=True,
                        help="root directory of wav files")
    parser.add_argument('--out_path', type=str, required=True,
                        help="root directory of wav files")
    parser.add_argument('-d', '--device', type=str, required=False,
                        help="device, cpu or cuda:0, cuda:1,...") 
    parser.add_argument('--sample_rate', type=int, required=False, default=16000,
                        help="Sample rate to convert")      
    parser.add_argument('-r', '--resample_mode', type=str, required=False,default='kaiser_best',
                        help="use kaiser_best for high-quality audio") 

    args = parser.parse_args()
    logger.info(f'using resample mode {args.resample_mode}')

    wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'),recursive=False)
    logger.info(f'{len(wav_files)} found in {args.wav_path}')

    # Create all folders
    os.makedirs(args.out_path, exist_ok=True)
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        # load wav file with librosa
        # conver mono channel
        # resample to config setting
        wav,sr = librosa.load(wavpath,sr=args.sample_rate,res_type=args.resample_mode, mono=True)
        # normalize
        wav = librosa.util.normalize(wav)
        # save to file
        outpath = os.path.join(args.out_path, os.path.basename(wavpath))
        sf.write(outpath, wav, args.sample_rate, subtype='PCM_16')
