# Copyright 2024 Adobe. All rights reserved.

#%%
import glob
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np


# %%
class MomentsDataset(Dataset):
    def __init__(self, videos_folder, num_frames, samples_per_video, frame_size=512) -> None:
        super().__init__()
        
        self.videos_paths = glob.glob(f'{videos_folder}/*mp4')
        self.resize = torchvision.transforms.Resize(size=frame_size)
        self.center_crop = torchvision.transforms.CenterCrop(size=frame_size)
        self.num_samples_per_video = samples_per_video
        self.num_frames = num_frames

    def __len__(self):
        return len(self.videos_paths) * self.num_samples_per_video
    
    def __getitem__(self, idx):
        video_idx = idx // self.num_samples_per_video
        video_path = self.videos_paths[video_idx]
        
        try:
            start_idx = np.random.randint(0, 20)
            
            unsampled_video_frames, audio_frames, info = torchvision.io.read_video(video_path,output_format="TCHW")
            sampled_indices = torch.tensor(np.linspace(start_idx, len(unsampled_video_frames)-1, self.num_frames).astype(int))
            sampled_frames = unsampled_video_frames[sampled_indices]
            processed_frames = []

            for frame in sampled_frames:
                resized_cropped_frame = self.center_crop(self.resize(frame))
                processed_frames.append(resized_cropped_frame)
            frames = torch.stack(processed_frames, dim=0)
            frames = frames.float() / 255.0
        except Exception as e:
            print('oops', e)
            rand_idx = np.random.randint(0, len(self))
            return self.__getitem__(rand_idx)
        
        out_dict = {'frames': frames,
         'caption': 'none',
         'keywords': 'none'}
        
        return out_dict
        

