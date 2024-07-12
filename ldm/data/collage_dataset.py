# Copyright 2024 Adobe. All rights reserved.

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import glob
import torchvision
from PIL import Image
import time
import os
import tqdm
from torch.utils.data import Dataset
import pathlib
import cv2
from PIL import Image
import os
import json
import albumentations as A

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        # transform_list += [torchvision.transforms.Normalize((0.0, 0.0, 0.0),
        #                                         (10.0, 10.0, 10.0))]
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = [torchvision.transforms.Resize((224,224))]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_dino(normalize=True, toTensor=True):
    transform_list = [torchvision.transforms.Resize((224,224))]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [lambda x: 255.0 * x[:3],
                           torchvision.transforms.Normalize(
                            mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                            )]
    return torchvision.transforms.Compose(transform_list)

def crawl_folders(folder_path):
    # glob crawl
    all_files = []
    folders = glob.glob(f'{folder_path}/*')
    
    for folder in folders:
        src_paths = glob.glob(f'{folder}/src_*png')
        all_files.extend(src_paths)
    return all_files

def get_grid(size):
    y = np.repeat(np.arange(size)[None, ...], size)
    y = y.reshape(size, size)
    x = y.transpose()
    out = np.stack([y,x], -1)
    return out
    

class CollageDataset(Dataset):
    def __init__(self, split_files, image_size, embedding_type, warping_type, blur_warped=False):
        self.size = image_size
        # depends on the embedding type
        if embedding_type == 'clip':
            self.get_embedding_vector = get_tensor_clip()
        elif embedding_type == 'dino':
            self.get_embedding_vector = get_tensor_dino()
        self.get_tensor = get_tensor()
        self.resize =  torchvision.transforms.Resize(size=(image_size, image_size))
        self.to_mask_tensor = get_tensor(normalize=False)
        
        self.src_paths = crawl_folders(split_files)
        print('current split size', len(self.src_paths))
        print('for dir', split_files)
        
        assert warping_type in ['collage', 'flow', 'mix']
        self.warping_type = warping_type
        
        self.mask_threshold = 0.85
        
        self.blur_t = torchvision.transforms.GaussianBlur(kernel_size=51, sigma=20.0)
        self.blur_warped = blur_warped
        
        # self.save_folder = '/mnt/localssd/collage_out'
        # os.makedirs(self.save_folder, exist_ok=True)
        self.save_counter = 0
        self.save_subfolder = None
    
    def __len__(self):
        return len(self.src_paths)
    
    
    def __getitem__(self, idx, depth=0):
        
        if self.warping_type == 'mix':
            # randomly sample
            warping_type = np.random.choice(['collage', 'flow'])
        else:
            warping_type = self.warping_type
        
        src_path = self.src_paths[idx]
        tgt_path = src_path.replace('src_', 'tgt_')
        
        if warping_type == 'collage':
            warped_path = src_path.replace('src_', 'composite_')
            mask_path = src_path.replace('src_', 'composite_mask_')
            corresp_path = src_path.replace('src_', 'composite_grid_')
            corresp_path = corresp_path.split('.')[0]
            corresp_path += '.npy'
        elif warping_type == 'flow':
            warped_path = src_path.replace('src_', 'flow_warped_')
            mask_path = src_path.replace('src_', 'flow_mask_')
            corresp_path = src_path.replace('src_', 'flow_warped_grid_')
            corresp_path = corresp_path.split('.')[0]
            corresp_path += '.npy'
        else:
            raise ValueError
        
        # load reference image, warped image, and target GT image
        reference_img = Image.open(src_path).convert('RGB')
        gt_img = Image.open(tgt_path).convert('RGB')
        warped_img = Image.open(warped_path).convert('RGB')
        warping_mask = Image.open(mask_path).convert('RGB')
        
        # resize all 
        reference_img = self.resize(reference_img)
        gt_img = self.resize(gt_img)
        warped_img = self.resize(warped_img)
        warping_mask = self.resize(warping_mask)

        
        # NO CROPPING PLEASE. ALL INPUTS ARE 512X512
        # Random crop
        # i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        #     reference_img, output_size=(512, 512))
        
        # reference_img = torchvision.transforms.functional.crop(reference_img, i, j, h, w)
        # gt_img = torchvision.transforms.functional.crop(gt_img, i, j, h, w)
        # warped_img = torchvision.transforms.functional.crop(warped_img, i, j, h, w)
        # # TODO start using the warping mask
        # warping_mask = torchvision.transforms.functional.crop(warping_mask, i, j, h, w)
        
        grid_transformed = torch.tensor(np.load(corresp_path))
        # grid_transformed = torchvision.transforms.functional.crop(grid_transformed, i, j, h, w)
        
        
        
        # reference_t = to_tensor(reference_img)
        gt_t = self.get_tensor(gt_img)
        warped_t = self.get_tensor(warped_img)
        warping_mask_t = self.to_mask_tensor(warping_mask)
        clean_reference_t = self.get_tensor(reference_img)
        # compute error to generate mask
        blur_t = torchvision.transforms.GaussianBlur(kernel_size=(11,11), sigma=5.0)
        
        reference_clip_img = self.get_embedding_vector(reference_img)        
        
        mask = torch.ones_like(gt_t)[:1]
        warping_mask_t = warping_mask_t[:1]
        
        good_region = torch.mean(warping_mask_t)
        # print('good region', good_region)
        # print('good region frac', good_region)
        if good_region < 0.4 and depth < 3:
            # example too hard, sample something else
            # print('bad image, resampling..')
            rand_idx = np.random.randint(len(self.src_paths))
            return self.__getitem__(rand_idx, depth+1)
        
        # if mask is too large then ignore
        
        # #gaussian inpainting now
        missing_mask = warping_mask_t[0] < 0.5
        
        
        reference = (warped_t.clone() + 1)  / 2.0
        ref_cv = torch.moveaxis(reference, 0, -1).cpu().numpy()
        ref_cv = (ref_cv * 255).astype(np.uint8)
        cv_mask = missing_mask.int().squeeze().cpu().numpy().astype(np.uint8)
        kernel = np.ones((7,7))
        dilated_mask = cv2.dilate(cv_mask, kernel)
        # cv_mask = np.stack([cv_mask]*3, axis=-1)
        dst = cv2.inpaint(ref_cv,dilated_mask,5,cv2.INPAINT_NS)
        
        mask_resized = torchvision.transforms.functional.resize(warping_mask_t, (64,64))
        # print(mask_resized)
        size=512
        grid_np = (get_grid(size) / size).astype(np.float16)# 512 x 512 x 2
        grid_t = torch.tensor(grid_np).moveaxis(-1, 0) # 512 x 512 x 2
        grid_resized = torchvision.transforms.functional.resize(grid_t, (64,64)).to(torch.float16)
        changed_pixels = torch.logical_or((torch.abs(grid_resized - grid_transformed)[0] > 0.04) , (torch.abs(grid_resized - grid_transformed)[1] > 0.04))
        changed_pixels = changed_pixels.unsqueeze(0)
        # changed_pixels = torch.logical_and(changed_pixels, (mask_resized >= 0.3))
        changed_pixels = changed_pixels.float()
        
        inpainted_warped = (torch.tensor(dst).moveaxis(-1, 0).float() / 255.0) * 2.0 - 1.0
        
        if self.blur_warped:
            inpainted_warped= self.blur_t(inpainted_warped)
        
        out = {"GT": gt_t,"inpaint_image": inpainted_warped,"inpaint_mask": warping_mask_t, "ref_imgs": reference_clip_img, "clean_reference": clean_reference_t, 'grid_transformed': grid_transformed, "changed_pixels": changed_pixels}
        # out = {"GT": gt_t,"inpaint_image": inpainted_warped * 0.0,"inpaint_mask": torch.ones_like(warping_mask_t), "ref_imgs": reference_clip_img * 0.0, "clean_reference": gt_t, 'grid_transformed': grid_transformed, "changed_pixels": changed_pixels}
        # out = {"GT": gt_t,"inpaint_image": inpainted_warped * 0.0,"inpaint_mask": warping_mask_t, "ref_imgs": reference_clip_img * 0.0, "clean_reference": clean_reference_t, 'grid_transformed': grid_transformed, "changed_pixels": changed_pixels}

        # out = {"GT": gt_t,"inpaint_image": warped_t,"inpaint_mask": warping_mask_t, "ref_imgs": reference_clip_img, "clean_reference": clean_reference_t, 'grid_transformed': grid_transformed, 'inpainted': inpainted_warped}
        # out_half = {key: out[key].half() for key in out}
        # if self.save_counter < 50:
        #     save_path = f'{self.save_folder}/output_{time.time()}.pt'
        #     torch.save(out, save_path)
        #     self.save_counter += 1
        
        return out

        
        
    
