"""
This file defines the Ego4DSounds Dataset for loading and processing video and audio. It includes functionality
preprocessing and loading media files and debugging outputs for analysis.
"""
import os
import sys
import json
import pandas as pd
import random
import torch
import glob
from collections import defaultdict

from torch.utils.data import Dataset
import torch
import torchaudio
import decord
import noisereduce as nr
from PIL import Image
from moviepy.editor import *
from decord import AudioReader, VideoReader
from decord import cpu, gpu
import matplotlib.pyplot as plt
import numpy as np
import torchaudio.functional as F
from librosa.util import normalize
import logging
import types
from collections import defaultdict

decord.bridge.set_bridge('torch')

class Ego4DSounds(Dataset):
    """Dataset class for handling video and audio data from the Ego4D dataset.
    This class supports loading and preprocessing of video and audio clips based on metadata.
    """

    def __init__(self,
                split,
                dataset_name,
                video_params,
                audio_params,
                data_dir,
                metadata_file=None,
                seed=0,
                metadata_dir=None, # for backward compatibility
                args=None, # for backward compatibility
                ):
        self.dataset_name = dataset_name
        self.video_params = video_params
        self.audio_params = audio_params
        self.data_dir = os.path.expandvars(data_dir)  # check for environment variables
        self.split = split
        self.metadata = pd.read_csv(metadata_file, sep='\t', on_bad_lines='warn')
        print(self.metadata.head())
        self.seed = seed
            
    def set_args(self, args):
        self.args = args # getting args directly from main script
        
    def get_id(self, sample):
        if 'narration_source' in sample and 'narration_ind' in sample:
            return sample['video_uid'] + '_' + sample['narration_source'] + '_' + str(sample['narration_ind'])
        else:
            return sample['video_uid']

    def __len__(self):
        return len(self.metadata)

    @property
    def video_size(self):
        return (self.args.num_frames, self.video_params['input_res'], self.video_params['input_res'], 3)

    @property
    def spec_size(self):
        return (self.audio_params['input_fdim'], self.audio_params['input_tdim'])

    @property
    def waveform_size(self):
        return (1, int(self.audio_params['sample_rate'] * self.audio_params['duration']))

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp = os.path.join(self.data_dir, sample['clip_file'])
        # logging.info(f"loading video {video_fp}")
        text = sample['clip_text']
        clip_id = self.get_id(sample)
        
        video = self.load_video(video_fp, num_frames=self.args.num_frames)
        waveform = self.load_audio(video_fp)
        
        return {'video': video, 'wav': waveform, 'clip_id': clip_id,}

    def load_video(self, video_fp, num_frames):
        video_size= (num_frames, self.video_params['input_res'], self.video_params['input_res'], 3)
        try:
            vr = VideoReader(video_fp, ctx=cpu(0))
            frame_indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
            imgs = vr.get_batch(frame_indices).float()
        except Exception as e:
            print('failed to load video, use black image instead', e)
            imgs = torch.zeros(video_size)

        imgs = (imgs / 255.0).permute(0, 3, 1, 2)  # [T, H, W, C] ---> [T, C, H, W]
        return imgs

    def load_audio(self, audio_fp):
        try:
            ar = AudioReader(audio_fp, ctx=cpu(0), sample_rate=16000)
            waveform = ar[:]
            if waveform.shape[1] > self.waveform_size[1]:
                waveform = waveform[:, :self.waveform_size[1]]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.waveform_size[1] - waveform.shape[1]))
        except Exception as e:
            print(f'Exception while reading audio file {audio_fp} with {e}')
            waveform = torch.zeros(self.waveform_size)
            
        return waveform[0]
    
class ego4dsounds_train(Ego4DSounds):
    def __init__(self, dataset_cfg):
        super().__init__(split="train", **dataset_cfg)

class ego4dsounds_validation(Ego4DSounds):
    def __init__(self, dataset_cfg):
        super().__init__(split="validation", **dataset_cfg)

class ego4dsounds_test(Ego4DSounds):
    def __init__(self, dataset_cfg):
        super().__init__(split="test", **dataset_cfg)

# debugging
if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    import matplotlib.pyplot as plt
    import soundfile as sf
    import time
    import numpy as np
    import logging
    
    # faciliate logging/debugging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    np.random.seed(seed=int(time.time())) 
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.seed = 0
    args.transform = True
    args.seg_len = 16000
    args.lowest_k = 1
    args.num_frames = 16
    
    kwargs = dict(
        split="train",
        dataset_name="ego4dsounds",
        data_dir="data/ego4dsounds_224p_test", # Ego4DSounds clips
        metadata_file="test_clips_11k.csv",
        video_params={
            "input_res": 224,
            "loading": "lax",
        },
        audio_params={
            "sample_rate": 16000,
            "duration": 3,
            "input_fdim": 128,
            "input_tdim": 196,
        },
        args=args
    )
    
    dataset = Ego4DSounds(**kwargs)
    dataset.set_args(args)
    num_video = len(dataset)
    print('Total number of videos clips: {}'.format(num_video))

    # randomly sample 10 videos
    indices = np.random.choice(num_video, 10)
    # indices = range(num_video)

    output_dir = 'debug_audio_video'
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    print('Total number of videos clips: {}'.format(num_video))

    # save audio and video to disk for debugging
    for i in tqdm(indices):
        item = dataset[i]
        video = item['video']
        audio = item['wav']
        clip_id = item['clip_id']

        print(audio.shape, video.shape)

        # save audio
        sf.write(f"{output_dir}/{clip_id}.wav", audio.numpy(), 16000)
        
        # concate frames horizontally and save image
        video = [img for img in video.permute(0, 2, 3, 1).numpy()]
        video = np.concatenate(video, axis=1)
        plt.imsave(f'{output_dir}/{clip_id}_video.png', video)
    
    print(f'Time taken: {time.time() - start}')
    print(f'Average time per video: {(time.time() - start) / len(indices)}')