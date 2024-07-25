# Copyright 2024 Adobe. All rights reserved.
#%%
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
import time
from pytorch_lightning import seed_everything
from torch import autocast
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.transforms import Resize
import argparse
import os
import pathlib
import glob
import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_model_from_config(config):
    model = instantiate_from_config(config.model)

    model.cuda()
    model.eval()
    return model


def get_model(config_path, ckpt_path, pretrained_sd_path):
    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config)
    model.load_state_dict(torch.load(pretrained_sd_path,map_location='cpu')['state_dict'],strict=False)

    
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    wrapped_state_dict = pl_sd #self.lightning_module.trainer.model.state_dict()
    new_sd = {k.replace("_forward_module.", ""): wrapped_state_dict[k] for k in wrapped_state_dict}

    m, u = model.load_state_dict(new_sd, strict=False)
    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)

    
    model = model.to(device)
    return model

def file_exists(path):
    """ Check if a file exists and is not a directory. """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file.")
    return path

def parse_arguments():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Process images based on provided paths.")
    parser.add_argument("--pretrained_sd", type=file_exists, required=True, help="Path to the SD1.4 pretrained checkpoint")
    parser.add_argument("--learned_params", type=file_exists, required=True, help="Path to the MagicFixup learned parameters.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the full model state dict")

    return parser.parse_args()

def main():
    args = parse_arguments()
    model = get_model('configs/collage_mix_train.yaml',args.learned_params, args.pretrained_sd)
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main()