# Copyright 2024 Adobe. All rights reserved.

#%%
from torchvision.transforms import ToPILImage
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import cv2
import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.utils import save_image
import time
import os
import sys
import pathlib
from torchvision.utils import flow_to_image
from torch.utils.data import DataLoader
from einops import rearrange
# %matplotlib inline
from kornia.filters.median import MedianBlur
median_filter = MedianBlur(kernel_size=(15,15))
from moments_dataset import MomentsDataset

try:
    from processing_utils import aggregate_frames
    import processing_utils
except Exception as e:
    print(e)
    print('process failed')
    exit()




import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

# %%

def load_image(img_path, resize_size=None,crop_size=None):

    img1_pil = Image.open(img_path)
    img1_frames = torchvision.transforms.functional.pil_to_tensor(img1_pil)
    
    if resize_size:
        img1_frames = torchvision.transforms.functional.resize(img1_frames, resize_size)

    if crop_size:
        img1_frames = torchvision.transforms.functional.center_crop(img1_frames, crop_size)

    img1_batch = torch.unsqueeze(img1_frames, dim=0)
    
    return img1_batch

def get_grid(size):
    y = np.repeat(np.arange(size)[None, ...], size)
    y = y.reshape(size, size)
    x = y.transpose()
    out = np.stack([y,x], -1)
    return out        

def collage_from_frames(frames_t):
    # decide forward or backward
    if np.random.randint(0, 2) == 0:
        # flip
        frames_t = frames_t.flip(0)
    
    # decide how deep you would go
    tgt_idx_guess = np.random.randint(1, min(len(frames_t), 20))
    tgt_idx = 1
    pairwise_flows = []
    flow = None
    init_time = time.time()
    unsmoothed_agg = None
    for cur_idx in range(1, tgt_idx_guess+1):
        # cur_idx = i+1
        cur_flow, pairwise_flows = aggregate_frames(frames_t[:cur_idx+1] , pairwise_flows, unsmoothed_agg) # passing pairwise flows for efficiency
        unsmoothed_agg = cur_flow.clone()
        agg_cur_flow = median_filter(cur_flow)
        
        flow_norm = torch.norm(agg_cur_flow.squeeze(), dim=0).flatten()
        # flow_10 = np.percentile(flow_norm.cpu().numpy(), 10)
        flow_90 = np.percentile(flow_norm.cpu().numpy(), 90)
        
        # flow_10 = np.percentile(flow_norm.cpu().numpy(), 10)
        flow_90 = np.percentile(flow_norm.cpu().numpy(), 90)
        flow_95 = np.percentile(flow_norm.cpu().numpy(), 95)
        
        if cur_idx == 5: # if still small flow then drop
            if flow_95 < 20.0:
                # no motion in the frame. skip
                print('flow is tiny :(')
                return None
        
        if cur_idx == tgt_idx_guess-1: # if still small flow then drop
            if flow_95 < 50.0:
                # no motion in the frame. skip
                print('flow is tiny :(')
                return None
        
        if flow is None: # means first iter
            if flow_90 < 1.0:
                # no motion in the frame. skip
                return None
            flow = agg_cur_flow
        
        if flow_90 <= 300: # maybe should increase this part
            # update idx
            tgt_idx = cur_idx
            flow = agg_cur_flow
        else:
            break
    final_time = time.time()
    print('time guessing idx', final_time - init_time)
    
    _, flow_warping_mask = processing_utils.forward_warp(frames_t[0], frames_t[tgt_idx], flow, grid=None, alpha_mask=None)
    flow_warping_mask = flow_warping_mask.squeeze().numpy() > 0.5
    
    if np.mean(flow_warping_mask) < 0.6:
        return
        
    
    src_array = frames_t[0].moveaxis(0, -1).cpu().numpy() * 1.0
    init_time = time.time()
    depth = get_depth_from_array(frames_t[0])
    finish_time = time.time()
    print('time getting depth', finish_time - init_time)
    # flow, pairwise_flows = aggregate_frames(frames_t)
    # agg_flow = median_filter(flow)
    
    src_array_uint = src_array * 255.0
    src_array_uint = src_array_uint.astype(np.uint8)
    segments = processing_utils.mask_generator.generate(src_array_uint)
    
    size = src_array.shape[1]
    grid_np = get_grid(size).astype(np.float16) / size # 512 x 512 x 2get
    grid_t = torch.tensor(grid_np).moveaxis(-1, 0) # 512 x 512 x 2
    
    
    collage, canvas_alpha, lost_alpha = collage_warp(src_array, flow.squeeze(), depth, segments, grid_array=grid_np)
    lost_alpha_t = torch.tensor(lost_alpha).squeeze().unsqueeze(0)
    warping_alpha = (lost_alpha_t < 0.5).float()
    
    rgb_grid_splatted, actual_warped_mask = processing_utils.forward_warp(frames_t[0], frames_t[tgt_idx], flow, grid=grid_t, alpha_mask=warping_alpha)
    

    # basic blending now
    # print('rgb grid splatted', rgb_grid_splatted.shape)
    warped_src = (rgb_grid_splatted * actual_warped_mask).moveaxis(0, -1).cpu().numpy()
    canvas_alpha_mask = canvas_alpha == 0.0
    collage_mask = canvas_alpha.squeeze() + actual_warped_mask.squeeze().cpu().numpy()
    collage_mask = collage_mask > 0.5
    
    composite_grid = warped_src * canvas_alpha_mask + collage
    rgb_grid_splatted_np = rgb_grid_splatted.moveaxis(0, -1).cpu().numpy()
    
    return frames_t[0], frames_t[tgt_idx], rgb_grid_splatted_np, composite_grid, flow_warping_mask, collage_mask

def collage_warp(rgb_array, flow, depth, segments, grid_array):
    avg_depths = []
    avg_flows = []
    
    # src_array = src_array.moveaxis(-1, 0).cpu().numpy() #np.array(Image.open(src_path).convert('RGB')) / 255.0
    src_array = np.concatenate([rgb_array, grid_array], axis=-1)
    canvas = np.zeros_like(src_array)
    canvas_alpha = np.zeros_like(canvas[...,-1:]).astype(float)
    lost_regions = np.zeros_like(canvas[...,-1:]).astype(float)
    z_buffer = np.ones_like(depth)[..., None] * -1.0
    unsqueezed_depth = depth[..., None]
    
    affine_transforms = []
    
    filtered_segments = []
    for segment in segments:
        if segment['area'] > 300:
            filtered_segments.append(segment)
    
    for segment in filtered_segments:
        seg_mask = segment['segmentation']
        avg_flow = torch.mean(flow[:, seg_mask],dim=1)
        avg_flows.append(avg_flow)
        # median depth (conversion from disparity)
        avg_depth = torch.median(1.0 / (depth[seg_mask] + 1e-6))
        avg_depths.append(avg_depth)
        
        all_y, all_x = np.nonzero(segment['segmentation'])
        rand_indices = np.random.randint(0, len(all_y), size=50)
        rand_x = [all_x[i] for i in rand_indices]
        rand_y = [all_y[i] for i in rand_indices]

        src_pairs = [(x, y) for x, y in zip(rand_x, rand_y)]
        # tgt_pairs = [(x + w, y) for x, y in src_pairs]
        tgt_pairs = []
        # print('estimating affine') # TODO this can be faster
        for i in range(len(src_pairs)):
            x, y = src_pairs[i]
            dx, dy = flow[:, y, x]
            tgt_pairs.append((x+dx, y+dy))
        
        # affine_trans, inliers = cv2.estimateAffine2D(np.array(src_pairs).astype(np.float32), np.array(tgt_pairs).astype(np.float32))
        affine_trans, inliers = cv2.estimateAffinePartial2D(np.array(src_pairs).astype(np.float32), np.array(tgt_pairs).astype(np.float32))
        # print('num inliers', np.sum(inliers))
        # # print('num inliers', np.sum(inliers))
        affine_transforms.append(affine_trans)
        
    depth_sorted_indices = np.arange(len(avg_depths))
    depth_sorted_indices = sorted(depth_sorted_indices, key=lambda x: avg_depths[x])
    # sorted_masks = []
    # print('warping stuff')
    for idx in depth_sorted_indices:
        # sorted_masks.append(mask[idx])        
        alpha_mask = filtered_segments[idx]['segmentation'][..., None] * (lost_regions < 0.5).astype(float)
        src_rgba = np.concatenate([src_array, alpha_mask, unsqueezed_depth], axis=-1)
        warp_dst = cv2.warpAffine(src_rgba, affine_transforms[idx], (src_array.shape[1], src_array.shape[0]))
        warped_mask = warp_dst[..., -2:-1] # this is warped alpha
        warped_depth = warp_dst[..., -1:]
        warped_rgb = warp_dst[...,:-2]
        
        good_z_region = warped_depth > z_buffer
        
        warped_mask = np.logical_and(warped_mask > 0.5, good_z_region).astype(float)
        
        kernel = np.ones((3,3), float)
        # print('og masked shape', warped_mask.shape)
        # warped_mask = cv2.erode(warped_mask,(5,5))[..., None]
        # print('eroded masked shape', warped_mask.shape)
        canvas_alpha += cv2.erode(warped_mask,kernel)[..., None]
        
        lost_regions += alpha_mask
        canvas = canvas * (1.0 - warped_mask) + warped_mask * warped_rgb # TODO check if need to dialate here
        z_buffer = z_buffer * (1.0 - warped_mask) + warped_mask * warped_depth # TODO check if need to dialate here    # print('max lost region', np.max(lost_regions))
    return canvas, canvas_alpha, lost_regions

def get_depth_from_array(img_t):
    img_arr = img_t.moveaxis(0, -1).cpu().numpy() * 1.0
    # print(img_arr.shape)
    img_arr *= 255.0
    img_arr = img_arr.astype(np.uint8)
    input_batch = processing_utils.depth_transform(img_arr).cuda()

    with torch.no_grad():
        prediction = processing_utils.midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_arr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu()
    return output


# %%

def main():
    print('starting main')
    video_folder = './example_videos'
    save_dir = pathlib.Path('./processed_data')
    process_video_folder(video_folder, save_dir)
        
def process_video_folder(video_folder, save_dir):
    all_counter = 0
    success_counter = 0

    # save_folder = pathlib.Path('/dev/shm/processed')        
    # save_dir = save_folder / foldername #pathlib.Path('/sensei-fs/users/halzayer/collage2photo/testing_partitioning_dilate_extreme')
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = MomentsDataset(videos_folder=video_folder, num_frames=20, samples_per_video=5)
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataset)//batch_size):
            frames_to_visualize = batch["frames"]
            bs = frames_to_visualize.shape[0]
            
            for j in range(bs):
                frames = frames_to_visualize[j]
                caption = batch["caption"][j]

                collage_init_time = time.time()
                out = collage_from_frames(frames)
                collage_finish_time = time.time()
                print('collage processing time', collage_finish_time - collage_init_time)
                all_counter += 1
                if out is not None:
                    src_image, tgt_image, splatted, collage, flow_mask, collage_mask = out
                    
                    splatted_rgb = splatted[...,:3]
                    splatted_grid = splatted[...,3:].astype(np.float16)

                    collage_rgb = collage[...,:3]
                    collage_grid = collage[...,3:].astype(np.float16)
                    success_counter += 1
                else:
                    continue

                id_str = f'{success_counter:08d}'

                src_path = str(save_dir / f'src_{id_str}.png')
                tgt_path = str(save_dir / f'tgt_{id_str}.png')
                flow_warped_path = str(save_dir / f'flow_warped_{id_str}.png')
                composite_path = str(save_dir / f'composite_{id_str}.png')
                flow_mask_path = str(save_dir / f'flow_mask_{id_str}.png')
                composite_mask_path = str(save_dir / f'composite_mask_{id_str}.png')
                
                flow_grid_path = str(save_dir / f'flow_warped_grid_{id_str}.npy')
                composite_grid_path = str(save_dir / f'composite_grid_{id_str}.npy')
                
                save_image(src_image, src_path)
                save_image(tgt_image, tgt_path)
                
                collage_pil = Image.fromarray((collage_rgb * 255).astype(np.uint8))
                collage_pil.save(composite_path)
                
                splatted_pil = Image.fromarray((splatted_rgb * 255).astype(np.uint8))
                splatted_pil.save(flow_warped_path)
                
                flow_mask_pil = Image.fromarray((flow_mask.astype(float) * 255).astype(np.uint8))
                flow_mask_pil.save(flow_mask_path)
                
                composite_mask_pil = Image.fromarray((collage_mask.astype(float) * 255).astype(np.uint8))
                composite_mask_pil.save(composite_mask_path)
                
                splatted_grid_t = torch.tensor(splatted_grid).moveaxis(-1, 0)
                splatted_grid_resized = torchvision.transforms.functional.resize(splatted_grid_t, (64,64))
                
                collage_grid_t = torch.tensor(collage_grid).moveaxis(-1, 0)
                collage_grid_resized = torchvision.transforms.functional.resize(collage_grid_t, (64,64))
                np.save(flow_grid_path, splatted_grid_resized.cpu().numpy())
                np.save(composite_grid_path, collage_grid_resized.cpu().numpy())

                
                del out
                del splatted_grid
                del collage_grid
                del frames

            del frames_to_visualize
                        
            
   
#%%

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        print('process failed')

