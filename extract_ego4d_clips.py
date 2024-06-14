"""
This script extracts Ego4DSounds clips from the Ego4D dataset using metadata files. It extracts clips from longer videos, resizes 
them, and saves them to the specified output directory.
"""

import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_clip_from_video(csv_data, output_dir, video_lengths):
    print(csv_data.head(), output_dir)
    
    # iterate over all clips
    for i, row in tqdm(csv_data.iterrows(), total=len(csv_data)):
        video_uid = row['video_uid']
        clip_time = row['narration_time']
        
        start = clip_time - 1.5
        end = clip_time + 1.5
        if start < 0:
            start = 0
            end = 3
        if end > video_lengths[video_uid]:
            end = video_lengths[video_uid]
            start = end - 3
        clip_file = row['clip_file']
        
        # path for Ego4D videos
        video_file = '/vision/vision_data/Ego4D/v1/videos_256p/{}.mp4'.format(video_uid)
        
        output_file = os.path.join(output_dir, clip_file).replace('.wav', '.mp4')
        if os.path.exists(output_file):
            continue
        else:
            os.system('''ffmpeg -i {} -ss {} -to {} -vf "scale=224:224" -ar 16000 -ac 1 {}'''
                      .format(video_file, start, end, output_file))

def extract_clip_from_video_multiprocess(csv_data, output_dir, video_lengths):
    num_worker = 10
    # split csv file into num_worker parts
    csv_data = np.array_split(csv_data, num_worker)
    
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        for i in range(num_worker):
            executor.submit(extract_clip_from_video, csv_data[i], output_dir, video_lengths)
    
    # wait for all processes to finish
    executor.shutdown(wait=True)

def main():
    output_dir = 'data/ego4dsounds_224p'
    
    # specify metadata file
    csv_data = pd.read_csv('test_clips_11k.csv', sep='\t')
    
    video_uids = csv_data['video_uid'].unique()
    for video_uid in video_uids:
        os.makedirs(os.path.join(output_dir, video_uid), exist_ok=True)

    video_lengths = {}
    metadata = json.load(open('ego4d.json'))
    for video_metadata in metadata['videos']:
        video_lengths[video_metadata['video_uid']] = video_metadata['duration_sec']
        
    num_worker = 20
    # split csv file into num_worker parts
    csv_data = np.array_split(csv_data, num_worker)
    
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        for i in range(num_worker):
            executor.submit(extract_clip_from_video, csv_data[i], output_dir, video_lengths)
            
    # extract_clip_from_video(csv_data[0], output_dir, video_lengths)

def slurm():
    num_node = 50
    output_dir = 'data/ego4dsounds_224p'
   
    # extract data with submitit
    import submitit
    executor = submitit.AutoExecutor(folder='submitit_logs')
    executor.update_parameters(
        timeout_min=60 * 72,
        slurm_partition='learnfair',
        gpus_per_node=0,
        cpus_per_task=20,
        nodes=num_node,
    )

    csv_data = pd.read_csv('test_clips_11k.csv', sep='\t')
    video_uids = csv_data['video_uid'].unique()
    for video_uid in video_uids:
        os.makedirs(os.path.join(output_dir, video_uid), exist_ok=True)
    
    video_lengths = {}
    metadata = json.load(open('ego4d.json'))
    for video_metadata in metadata['videos']:
        video_lengths[video_metadata['video_uid']] = video_metadata['duration_sec']
        
    # split csv file into num_worker parts
    csv_data = np.array_split(csv_data, num_node)
    with executor.batch():
        for i in range(num_node):
            executor.submit(extract_clip_from_video_multiprocess, csv_data[i], output_dir, video_lengths) 


if __name__ == '__main__':
    main()
    # slurm()