# Copyright 2024 Adobe. All rights reserved.

from run_magicfu import MagicFixup
import os
import pathlib
import torchvision
from torch import autocast
from PIL import Image
import gradio as gr
import numpy as np
import argparse


def sample(original_image, coarse_edit):
    to_tensor = torchvision.transforms.ToTensor()
    with autocast("cuda"):
        w, h = coarse_edit.size
        ref_image_t = to_tensor(original_image.resize((512,512))).half().cuda()
        coarse_edit_t = to_tensor(coarse_edit.resize((512,512))).half().cuda()
        # get mask from coarse
        coarse_edit_mask_t = to_tensor(coarse_edit.resize((512,512))).half().cuda()
        mask_t = (coarse_edit_mask_t[-1][None, None,...]).half() # do center crop
        coarse_edit_t_rgb = coarse_edit_t[:-1]
        
        out_rgb = magic_fixup.edit_image(ref_image_t, coarse_edit_t_rgb, mask_t, start_step=1.0, steps=50)
        output =  out_rgb.squeeze().cpu().detach().moveaxis(0, -1).float().numpy()
        output = (output * 255.0).astype(np.uint8)
        output_pil = Image.fromarray(output)
        output_pil = output_pil.resize((w, h))
        return output_pil

def file_exists(path):
    """ Check if a file exists and is not a directory. """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file.")
    return path

def parse_arguments():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Process images based on provided paths.")
    parser.add_argument("--checkpoint", type=file_exists, required=True, help="Path to the MagicFixup checkpoint file.")

    return parser.parse_args()

demo = gr.Interface(fn=sample, inputs=[gr.Image(type="pil", image_mode='RGB'), gr.Image(type="pil", image_mode='RGBA')], outputs=gr.Image(),
                    examples='examples')
    
if __name__ == "__main__":
    args = parse_arguments()

    # create magic fixup model
    magic_fixup = MagicFixup(model_path=args.checkpoint)
    demo.launch(share=True)   
