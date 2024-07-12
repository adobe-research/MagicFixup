import torch
import cv2
import numpy as np
import sys
import torchvision
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
sys.path.append('./softmax-splatting')
import softsplat


sam_checkpoint = "./sam_model/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# mask_generator = SamAutomaticMaskGenerator(sam,
#                                            crop_overlap_ratio=0.05,
#                                            box_nms_thresh=0.2,
#                                            points_per_side=32,
#                                            pred_iou_thresh=0.86,
#                                            stability_score_thresh=0.8,

#                                            min_mask_region_area=100,)
# mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(sam,
                                        #    box_nms_thresh=0.5,
                                        #    crop_overlap_ratio=0.75,
                                        #    min_mask_region_area=200,
                                           )

def get_mask(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    return masks

def get_mask_from_array(arr):
    return mask_generator.generate(arr)

# depth model

import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

# potentially downgrade this. just need rough depths. benchmark this
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas = torch.hub.load("/sensei-fs/users/halzayer/collage2photo/model_cache/intel-isl_MiDaS_master", model_type, source='local')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transforms = torch.hub.load("/sensei-fs/users/halzayer/collage2photo/model_cache/intel-isl_MiDaS_master", "transforms", source='local')

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    depth_transform = midas_transforms.dpt_transform
else:
    depth_transform = midas_transforms.small_transform

# img_path = '/sensei-fs/users/halzayer/valid/JPEGImages/45597680/00005.jpg'
def get_depth(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = depth_transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu()
    return output

def get_depth_from_array(img):
    input_batch = depth_transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu()
    return output


def load_image(img_path):
    img1_names = [img_path]

    img1_pil = [Image.open(fn) for fn in img1_names]
    img1_frames = [torchvision.transforms.functional.pil_to_tensor(fn) for fn in img1_pil]

    img1_batch = torch.stack(img1_frames)
    
    return img1_batch

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

print('created model')

def preprocess(img1_batch, img2_batch, size=[520,960], transform_batch=True):
    img1_batch = F.resize(img1_batch, size=size, antialias=False)
    img2_batch = F.resize(img2_batch, size=size, antialias=False)
    if transform_batch:
        return transforms(img1_batch, img2_batch)
    else:
        return img1_batch, img2_batch

def compute_flow(img_path_1, img_path_2):
    img1_batch_og, img2_batch_og = load_image(img_path_1), load_image(img_path_2)
    B, C, H, W = img1_batch_og.shape

    img1_batch, img2_batch = preprocess(img1_batch_og, img2_batch_og, transform_batch=False)
    img1_batch_t, img2_batch_t = transforms(img1_batch, img2_batch)

    # If you can, run this example on a GPU, it will be a lot faster.
    with torch.no_grad():
        list_of_flows = model(img1_batch_t.to(device), img2_batch_t.to(device))
        predicted_flows = list_of_flows[-1]
        # flows.append(predicted_flows)

        resized_flow = F.resize(predicted_flows, size=(H, W), antialias=False)
        
        _, _, flow_H, flow_W = predicted_flows.shape
        
        resized_flow[:,0] *= (W / flow_W)
        resized_flow[:,1] *= (H / flow_H)

    return resized_flow.detach().cpu().squeeze()

def compute_flow_from_tensors(img1_batch_og, img2_batch_og):
    if len(img1_batch_og.shape) < 4:
        img1_batch_og = img1_batch_og.unsqueeze(0)
    if len(img2_batch_og.shape) < 4:
        img2_batch_og = img2_batch_og.unsqueeze(0)
    
    B, C, H, W = img1_batch_og.shape
    img1_batch, img2_batch = preprocess(img1_batch_og, img2_batch_og, transform_batch=False)
    img1_batch_t, img2_batch_t = transforms(img1_batch, img2_batch)

    # If you can, run this example on a GPU, it will be a lot faster.
    with torch.no_grad():
        list_of_flows = model(img1_batch_t.to(device), img2_batch_t.to(device))
        predicted_flows = list_of_flows[-1]
        # flows.append(predicted_flows)

        resized_flow = F.resize(predicted_flows, size=(H, W), antialias=False)
        
        _, _, flow_H, flow_W = predicted_flows.shape
        
        resized_flow[:,0] *= (W / flow_W)
        resized_flow[:,1] *= (H / flow_H)

    return resized_flow.detach().cpu().squeeze()



# import run
backwarp_tenGrid = {}

def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[3], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[2], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################
def forward_splt(src, tgt, flow, partial=False):
    tenTwo = tgt.unsqueeze(0).cuda() #torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename='./images/one.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
    tenOne = src.unsqueeze(0).cuda() #torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename='./images/two.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
    tenFlow = flow.unsqueeze(0).cuda() #torch.FloatTensor(numpy.ascontiguousarray(run.read_flo('./images/flow.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()

    if not partial:
        tenMetric = torch.nn.functional.l1_loss(input=tenOne, target=backwarp(tenIn=tenTwo, tenFlow=tenFlow), reduction='none').mean([1], True)
    else:
        tenMetric = torch.nn.functional.l1_loss(input=tenOne[:,:3], target=backwarp(tenIn=tenTwo[:,:3], tenFlow=tenFlow[:,:3]), reduction='none').mean([1], True)
    # for intTime, fltTime in enumerate(np.linspace(0.0, 1.0, 11).tolist()):
    tenSoftmax = softsplat.softsplat(tenIn=tenOne, tenFlow=tenFlow , tenMetric=(-20.0 * tenMetric).clip(-20.0, 20.0), strMode='soft') # -20.0 is a hyperparameter, called 'alpha' in the paper, that could be learned using a torch.Parameter

    return tenSoftmax.cpu()


def aggregate_frames(frames, pairwise_flows=None, agg_flow=None):
    if pairwise_flows is None:
        # store pairwise flows
        pairwise_flows = []
    
    if agg_flow is None:
        start_idx = 0
    else:
        start_idx = len(pairwise_flows)
    
    og_image = frames[start_idx]
    prev_frame = og_image
    
    for i in range(start_idx, len(frames)-1):
        tgt_frame = frames[i+1]

        if i < len(pairwise_flows):
            flow = pairwise_flows[i]
        else:
            flow = compute_flow_from_tensors(prev_frame, tgt_frame)
            pairwise_flows.append(flow.clone())

        _, H, W = flow.shape
        B=1

        xx = torch.arange(0, W).view(1,-1).repeat(H,1)

        yy = torch.arange(0, H).view(-1,1).repeat(1,W)

        xx = xx.view(1,1,H,W).repeat(B,1,1,1)

        yy = yy.view(1,1,H,W).repeat(B,1,1,1)

        grid = torch.cat((xx,yy),1).float()

        flow = flow.unsqueeze(0)
        if agg_flow is None:
            agg_flow = torch.zeros_like(flow)
        
        vgrid = grid  + agg_flow
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1) - 1

        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1) - 1

        flow_out = torch.nn.functional.grid_sample(flow, vgrid.permute(0,2,3,1), 'nearest')

        agg_flow += flow_out
        

        # mask = forward_splt(torch.ones_like(og_image), torch.ones_like(og_image), agg_flow.squeeze()).squeeze()
        # blur_t = torchvision.transforms.GaussianBlur(kernel_size=(25,25), sigma=5.0)
        # warping_mask = (blur_t(mask)[0:1] > 0.8)
        # masks.append(warping_mask)
        prev_frame = tgt_frame
    
    return agg_flow, pairwise_flows #og_splatted_img, agg_flow, actual_warped_mask


def forward_warp(src_frame, tgt_frame, flow, grid=None, alpha_mask=None):
    if alpha_mask is None:
        alpha_mask = torch.ones_like(src_frame[:1])
        
    if grid is not None:
        src_list = [src_frame, grid, alpha_mask]
        tgt_list = [tgt_frame, grid, alpha_mask]
    else:
        src_list = [src_frame, alpha_mask]
        tgt_list = [tgt_frame, alpha_mask]
    
    og_image_padded = torch.concat(src_list, dim=0)
    tgt_frame_padded = torch.concat(tgt_list, dim=0)
    
    og_splatted_img = forward_splt(og_image_padded, tgt_frame_padded, flow.squeeze(), partial=True).squeeze()
    # print('og splatted image shape')
    # grid_transformed = og_splatted_img[3:-1]
    # print('grid transformed shape', grid_transformed)
    
    # grid *= grid_size
    # grid_transformed *= grid_size
    actual_warped_mask = og_splatted_img[-1:]
    splatted_rgb_grid = og_splatted_img[:-1]

    return splatted_rgb_grid, actual_warped_mask