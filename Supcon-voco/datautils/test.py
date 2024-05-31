from datautils.audio_augmentor import BackgroundNoiseAugmentor, PitchAugmentor, ReverbAugmentor, SpeedAugmentor, VolumeAugmentor, TelephoneEncodingAugmentor, GaussianAugmentor, CopyPasteAugmentor, BaseAugmentor
from datautils.audio_augmentor.utils import pydub_to_librosa, librosa_to_pydub
import os

ta = CopyPasteAugmentor({'aug_type':'copy_paste','shuffle_ratio':0.5, 'frame_size': 1600, 'output_path': './', 'out_format': 'wav'})
ta.load('/datab/Dataset/cnsl_real_fake_audio/in_the_wild/27877.wav')
ta.transform()
ta.save()