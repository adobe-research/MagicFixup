# Copyright 2024 Adobe. All rights reserved.

#%%
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from torch import autocast
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.transforms import Resize
import argparse
import os
import pathlib
import glob


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def fix_img(test_img):
    width, height = test_img.size
    if width != height:
        left = 0
        right = height
        bottom = height
        top = 0
        return test_img.crop((left, top, right, bottom))
    else:
        return test_img
# util funcs
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
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

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    transform_list += [
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(512) 
    ]
    return torchvision.transforms.Compose(transform_list)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images



def load_model_from_config(config, ckpt, verbose=False):
    model = instantiate_from_config(config.model)
    # print('NOTE: NO CHECKPOINT IS LOADED')
    
    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        # sd = pl_sd["state_dict"]
        
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    model.cuda()
    model.eval()
    return model


def get_model(config_path, ckpt_path):
    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config, None)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    
    m, u = model.load_state_dict(pl_sd, strict=True)
    if len(m) > 0:
        print("WARNING: missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)

    
    model = model.to(device)
    return model

def get_grid(size):
    y = np.repeat(np.arange(size)[None, ...], size)
    y = y.reshape(size, size)
    x = y.transpose()
    out = np.stack([y,x], -1)
    return out

def un_norm(x):
    return (x+1.0)/2.0

class MagicFixup:
    def __init__(self, model_path='/sensei-fs/users/halzayer/collage2photo/Paint-by-Example/official_checkpoint_image_attn_200k.pt'):
        self.model = get_model('configs/collage_mix_train.yaml',model_path)
  

    def edit_image(self, ref_image, coarse_edit, mask_tensor, start_step, steps):
            # essentially sample
            sampler = DDIMSampler(self.model)

            start_code = None
            
            transformed_grid = torch.zeros((2, 64, 64))

            self.model.model.og_grid = None
            self.model.model.transformed_grid = transformed_grid.unsqueeze(0).to(self.model.device)

            scale = 1.0
            C, f, H, W= 4, 8, 512, 512
            n_samples = 1
            ddim_steps = steps
            ddim_eta = 1.0
            step = start_step        

            with torch.no_grad():
                with autocast("cuda"):
                    with self.model.ema_scope():
                        image_tensor = get_tensor(toTensor=False)(coarse_edit)
                        
                        clean_ref_tensor = get_tensor(toTensor=False)(ref_image)
                        clean_ref_tensor = clean_ref_tensor.unsqueeze(0)

                        ref_tensor=get_tensor_dino(toTensor=False)(ref_image).unsqueeze(0)

                        b_mask = mask_tensor.cpu() < 0.5
                        
                        # inpainting
                        reference = un_norm(image_tensor)
                        reference = reference.squeeze()
                        ref_cv = torch.moveaxis(reference, 0, -1).cpu().numpy()
                        ref_cv = (ref_cv * 255).astype(np.uint8)

                        cv_mask = b_mask.int().squeeze().cpu().numpy().astype(np.uint8)
                        kernel = np.ones((7,7))
                        dilated_mask = cv2.dilate(cv_mask, kernel)

                        dst = cv2.inpaint(ref_cv,dilated_mask,3,cv2.INPAINT_NS)
                        # dst = inpaint.inpaint_biharmonic(ref_cv, dilated_mask, channel_axis=-1)
                        dst_tensor = torch.tensor(dst).moveaxis(-1, 0) / 255.0
                        image_tensor = (dst_tensor * 2.0) - 1.0
                        image_tensor = image_tensor.unsqueeze(0)
                        
                        ref_tensor = ref_tensor

                        inpaint_image = image_tensor#*mask_tensor

                        test_model_kwargs={}
                        test_model_kwargs['inpaint_mask']=mask_tensor.to(device)
                        test_model_kwargs['inpaint_image']=inpaint_image.to(device)
                        clean_ref_tensor = clean_ref_tensor.to(device)
                        ref_tensor=ref_tensor.to(device)
                        uc = None
                        if scale != 1.0:
                            uc = self.model.learnable_vector
                        c = self.model.get_learned_conditioning(ref_tensor.to(torch.float16))
                        c = self.model.proj_out(c)
                                            
                        z_inpaint = self.model.encode_first_stage(test_model_kwargs['inpaint_image'])
                        z_inpaint = self.model.get_first_stage_encoding(z_inpaint).detach()
                        
                        
                        z_ref = self.model.encode_first_stage(clean_ref_tensor)
                        z_ref = self.model.get_first_stage_encoding(z_ref).detach()
                        
                        test_model_kwargs['inpaint_image']=z_inpaint
                        test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])


                        shape = [C, H // f, W // f]

                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                            conditioning=c,
                                                            z_ref=z_ref,
                                                            batch_size=n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=ddim_eta,
                                                            x_T=start_code,
                                                            test_model_kwargs=test_model_kwargs,
                                                            x0=z_inpaint, 
                                                            x0_step=step,
                                                            ddim_discretize='uniform',
                                                            drop_latent_guidance=1.0
                                                            )                
                        

                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image=x_samples_ddim
                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)                
                        

                        return x_checked_image_torch
#%%


#%%
import time



# %%
def file_exists(path):
    """ Check if a file exists and is not a directory. """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file.")
    return path

def parse_arguments():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Process images based on provided paths.")
    parser.add_argument("--checkpoint", type=file_exists, required=True, help="Path to the MagicFixup checkpoint file.")
    parser.add_argument("--reference", type=file_exists, default='examples/fox_drinking_og.png', help="Path to the reference original image.")
    parser.add_argument("--edit", type=file_exists, default='examples/fox_drinking__edit__01.png', help="Path to the image edit. Make sure the alpha channel is set properly")
    parser.add_argument("--output-dir", type=str, default='./outputs', help="Path to the folder where to save the outputs")
    parser.add_argument("--samples", type=int, default=5, help="number of samples to output")
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()

    # create magic fixup model
    magic_fixup = MagicFixup(model_path=args.checkpoint)
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # run it here

    to_tensor = torchvision.transforms.ToTensor()



    ref_path = args.reference
    coarse_edit_path = args.edit
    mask_edit_path = coarse_edit_path

    edit_file_name = pathlib.Path(coarse_edit_path).stem
    save_pattern = f'{output_dir}/{edit_file_name}__sample__*.png'
    save_counter = len(glob.glob(save_pattern))

    all_rgbs = []
    for i in range(args.samples):
        with autocast("cuda"):
            ref_image_t = to_tensor(Image.open(ref_path).convert('RGB').resize((512,512))).half().cuda()
            coarse_edit_t = to_tensor(Image.open(coarse_edit_path).resize((512,512))).half().cuda()
            # get mask from coarse
            # mask_t = torch.ones_like(coarse_edit_t[-1][None, None,...])
            coarse_edit_mask_t = to_tensor(Image.open(mask_edit_path).resize((512,512))).half().cuda()
            # get mask from coarse
            mask_t = (coarse_edit_mask_t[-1][None, None,...]).half() # do center crop
            coarse_edit_t_rgb = coarse_edit_t[:-1]
            
            out_rgb = magic_fixup.edit_image(ref_image_t, coarse_edit_t_rgb, mask_t, start_step=1.0, steps=50)
            all_rgbs.append(out_rgb.squeeze().cpu().detach().float())
    
            save_path = f'{output_dir}/{edit_file_name}__sample__{save_counter:03d}.png'
            torchvision.utils.save_image(all_rgbs[i], save_path)
            save_counter += 1



if __name__ == "__main__":
    main()