import argparse
import math
import random
import os,time
import subprocess

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils
from tqdm import tqdm

from torch.autograd import Variable
import matplotlib as mlb

from PIL import Image, ImageDraw, ImageFont
import itertools
from tensorboardX import SummaryWriter
import random
import torch

import cv2
import glob
import numpy as np
import os
import face_alignment

from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage import io
import torchvision#.transforms.functional import perspective# as tvp
import torch
from torchvision import transforms, utils

from utils.morphology import dilation

import os
import torch
import torch.nn as nn
import time
import numpy as np

TAG_CHAR = np.array([202021.25], np.float32)


### python lib
import os, sys, random, math, cv2, pickle, subprocess
import numpy as np
from PIL import Image

### torch lib
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

### custom lib

try:
    from networks.resample2d_package.resample2d import Resample2d
except:
    from .networks.resample2d_package.resample2d import Resample2d

FLO_TAG = 202021.25
EPS = 1e-12

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


######################################################################################
##  Training utility
######################################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def normalize_ImageNet_stats(batch):

    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = (batch - mean) / std

    return batch_out


def img2tensor(img):

    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t

def tensor2img(img_t):

    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


def save_model(model, optimizer, opts):

    # save opts
    opts_filename = os.path.join(opts.model_dir, "opts.pth")
    print("Save %s" %opts_filename)
    with open(opts_filename, 'wb') as f:
        pickle.dump(opts, f)

    # serialize model and optimizer to dict
    state_dict = {
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }

    model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" %model.epoch)
    print("Save %s" %model_filename)
    torch.save(state_dict, model_filename)


def load_model(model, optimizer, opts, epoch):

    # load model
    model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" %epoch)
    print("Load %s" %model_filename)
    state_dict = torch.load(model_filename)
    
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])

    ### move optimizer state to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    model.epoch = epoch ## reset model epoch

    return model, optimizer



class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def create_data_loader(data_set, opts, mode):

    ### generate random index
    if mode == 'train':
        total_samples = opts.train_epoch_size * opts.batch_size
    else:
        total_samples = opts.valid_epoch_size * opts.batch_size

    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    ### generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    data_loader = DataLoader(dataset=data_set, num_workers=opts.threads, batch_size=opts.batch_size, sampler=sampler, pin_memory=True)

    return data_loader


def learning_rate_decay(opts, epoch):
    
    ###             1 ~ offset              : lr_init
    ###        offset ~ offset + step       : lr_init * drop^1
    ### offset + step ~ offset + step * 2   : lr_init * drop^2
    ###              ...
    
    if opts.lr_drop == 0: # constant learning rate
        decay = 0
    else:
        assert(opts.lr_step > 0)
        decay = math.floor( float(epoch) / opts.lr_step )
        decay = max(decay, 0) ## decay = 1 for the first lr_offset iterations

    lr = opts.lr_init * math.pow(opts.lr_drop, decay)
    lr = max(lr, opts.lr_init * opts.lr_min)

    return lr


def count_network_parameters(model):

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])

    return N



######################################################################################
##  Image utility
######################################################################################


def rotate_image(img, degree, interp=cv2.INTER_LINEAR):

    height, width = img.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, degree, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    img_out = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), flags=interp+cv2.WARP_FILL_OUTLIERS)
  
    return img_out


def numpy_to_PIL(img_np):

    ## input image is numpy array in [0, 1]
    ## convert to PIL image in [0, 255]

    img_PIL = np.uint8(img_np * 255)
    img_PIL = Image.fromarray(img_PIL)

    return img_PIL

def PIL_to_numpy(img_PIL):

    img_np = np.asarray(img_PIL)
    img_np = np.float32(img_np) / 255.0

    return img_np


def read_img(filename, grayscale=0):

    ## read image and convert to RGB in [0, 1]

    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = img[:, :, ::-1] ## BGR to RGB
    
    img = np.float32(img) / 255.0

    return img

def save_img(img, filename):

    print("Save %s" %filename)

    if img.ndim == 3:
        img = img[:, :, ::-1] ### RGB to BGR
    
    ## clip to [0, 1]
    img = np.clip(img, 0, 1)

    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)

    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


######################################################################################
##  Flow utility
######################################################################################

def read_flo(filename):

    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        
        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' %filename)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            #print 'Reading %d x %d flo file' % (w, h)
                
            data = np.fromfile(f, np.float32, count=2*w*h)

            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))

    return flow

def save_flo(flow, filename):

    with open(filename, 'wb') as f:

        tag = np.array([FLO_TAG], dtype=np.float32)

        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        tag.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)
    
def resize_flow(flow, W_out=0, H_out=0, scale=0):

    if W_out == 0 and H_out == 0 and scale == 0:
        raise Exception("(W_out, H_out) or scale should be non-zero")

    H_in = flow.shape[0]
    W_in = flow.shape[1]

    if scale == 0:
        y_scale = float(H_out) / H_in
        x_scale = float(W_out) / W_in
    else:
        y_scale = scale
        x_scale = scale

    flow_out = cv2.resize(flow, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)

    flow_out[:, :, 0] = flow_out[:, :, 0] * x_scale
    flow_out[:, :, 1] = flow_out[:, :, 1] * y_scale

    return flow_out


def rotate_flow(flow, degree, interp=cv2.INTER_LINEAR):
    
    ## angle in radian
    angle = math.radians(degree)

    H = flow.shape[0]
    W = flow.shape[1]

    #rotation_matrix = cv2.getRotationMatrix2D((W/2, H/2), math.degrees(angle), 1)
    #flow_out = cv2.warpAffine(flow, rotation_matrix, (W, H))
    flow_out = rotate_image(flow, degree, interp)
    
    fu = flow_out[:, :, 0] * math.cos(-angle) - flow_out[:, :, 1] * math.sin(-angle)
    fv = flow_out[:, :, 0] * math.sin(-angle) + flow_out[:, :, 1] * math.cos(-angle)

    flow_out[:, :, 0] = fu
    flow_out[:, :, 1] = fv

    return flow_out

def hflip_flow(flow):

    flow_out = cv2.flip(flow, flipCode=0)
    flow_out[:, :, 0] = flow_out[:, :, 0] * (-1)

    return flow_out

def vflip_flow(flow):

    flow_out = cv2.flip(flow, flipCode=1)
    flow_out[:, :, 1] = flow_out[:, :, 1] * (-1)

    return flow_out

def flow_to_rgb(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    #print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.float32(img) / 255.0


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def compute_flow_magnitude(flow):

    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag

def compute_flow_gradients(flow):

    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))
    
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


def detect_occlusion(fw_flow, bw_flow):
    
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1

    
    with torch.no_grad():

        ## convert to tensor
        fw_flow_t = img2tensor(fw_flow).cuda()
        bw_flow_t = img2tensor(bw_flow).cuda()

        ## warp fw-flow to img2
        flow_warping = Resample2d().cuda()
        fw_flow_w = flow_warping(fw_flow_t, bw_flow_t)
    
        ## convert to numpy array
        fw_flow_w = tensor2img(fw_flow_w)


    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5
    
    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = np.logical_or(mask1, mask2)
    occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[mask == 1] = 1

    return occlusion

######################################################################################
##  Other utility
######################################################################################

def save_vector_to_txt(matrix, filename):

    with open(filename, 'w') as f:

        print("Save %s" %filename)
        
        for i in range(matrix.size):
            line = "%f" %matrix[i]
            f.write("%s\n"%line)

def run_cmd(cmd):
    print(cmd)
    subprocess.call(cmd, shell=True)

def make_video(input_dir, img_fmt, video_filename, fps=24):

    cmd = "ffmpeg -y -loglevel error -framerate %s -i %s/%s -vcodec libx264 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" %s" \
            %(fps, input_dir, img_fmt, video_filename)

    run_cmd(cmd)


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


def add_texts_to_image_vertical(texts, pivot_images):
    images_height = pivot_images.height
    images_width = pivot_images.width

    text_height = 256 + 16 - images_height % 32
    num_images = len(texts)
    image_width = images_width // num_images
    text_image = Image.new('RGB', (images_width, text_height), (255, 255, 255))
    draw = ImageDraw.Draw(text_image)
    font_size = int(math.ceil(24 * image_width / 256))

    for i, text in enumerate(texts):
        font = ImageFont.truetype("./FreeSans.ttf", font_size)
        draw.text((image_width // 3 + i * image_width, text_height // 2), text, fill='black', anchor='ms', font=font)

    out_image = Image.new('RGB', (pivot_images.width, pivot_images.height + text_image.height))
    out_image.paste(text_image, (0, 0))
    out_image.paste(pivot_images, (0, text_image.height))
    return out_image


def concat_images_horizontally(*images: Image.Image):
    assert all(image.height == images[0].height for image in images)
    total_width = sum(image.width for image in images)

    new_im = Image.new(images[0].mode, (total_width, images[0].height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im

def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is  "+ str(time.time() - startTime_for_tictoc)+"  seconds")
        #str(time.time() - startTime_for tictoc)
    else:
        print("Toc: start time not set")

def warp(x,flo, return_mask=True):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)

    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask = mask.masked_fill_(mask < 0.999, 0)
    mask = mask.masked_fill_(mask > 0, 1)

    if return_mask:
        return output * mask, mask
    else:
        return output * mask
        
def back_warping(x,flo, return_mask=False):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)

    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask = mask.masked_fill_(mask < 0.999, 0)
    mask = mask.masked_fill_(mask > 0, 1)

    if return_mask:
        return output * mask, mask
    else:
        return output * mask


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)
        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.
        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut




def reproject(ori_frame, pro_frame, mask, coeff, crop_size, quad_0):

    ori_frame = ori_frame.squeeze()
    pro_frame = pro_frame.squeeze()
    mask = mask.squeeze(0)

    coeff = coeff.squeeze()
    
    pro_frame = transforms.functional_tensor.perspective(pro_frame, coeff)
    pro_frame = pro_frame[ :, :crop_size[0], :crop_size[1]]        

    pro_frame_bg = torch.zeros_like(ori_frame)+ori_frame

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = pro_frame
    else:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = pro_frame


    mask = transforms.functional_tensor.perspective(mask, coeff)
    mask = mask[ :, :crop_size[0], :crop_size[1]]  
    mask_bg = torch.zeros_like(ori_frame)

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        mask_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = mask
    else:
        mask_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = mask


    output = ori_frame*(1-mask_bg) + pro_frame_bg*mask_bg
    return output.unsqueeze(0).float(),mask_bg





def reproject_batch(ori_frames, pro_frames, masks, coeffs, crop_sizes, quad_0s):
    outputs = []
    for i in range(ori_frames.shape[0]):
        ori_frame = ori_frames[i]
        pro_frame = pro_frames[i]
        mask = masks[i]#.squeeze(0)
        coeff = coeffs[i]
        crop_size = [crop_sizes[0][i],crop_sizes[1][i]]
        quad_0 = [quad_0s[0][i],quad_0s[1][i],quad_0s[2][i],quad_0s[3][i]]

        # ori_frame = ori_frame.squeeze()
        # pro_frame = pro_frame.squeeze()
        # mask = mask.squeeze(0)
        # coeff = coeff.squeeze()
    
        pro_frame = transforms.functional_tensor.perspective(pro_frame, coeff)
        pro_frame = pro_frame[ :, :crop_size[0], :crop_size[1]]        

        pro_frame_bg = torch.zeros_like(ori_frame)+ori_frame

        if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
            pro_frame_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = pro_frame
        else:
            pro_frame_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = pro_frame


        mask = transforms.functional_tensor.perspective(mask, coeff)
        mask = mask[ :, :crop_size[0], :crop_size[1]]  
        mask_bg = torch.zeros_like(ori_frame)

        if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
            mask_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = mask
        else:
            mask_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = mask


        output = ori_frame*(1-mask_bg) + pro_frame_bg*mask_bg
        outputs.append(output)
        
    # return output.unsqueeze(0).float(),mask_bg
    return torch.stack(outputs).float(),mask_bg





def reproject_wi_dilate_erode(ori_frame, pro_frame, mask, coeff, crop_size, quad_0):

    ori_frame = ori_frame.squeeze()
    pro_frame = pro_frame.squeeze()
    mask_ = mask.squeeze(0)


    mask = mask_.cpu().numpy()
    mask = (mask.transpose(1,2,0) * 255).astype(np.uint8)
    mask1 = cv2.dilate(mask, np.ones((20,20)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    



    mask = mask_.cpu().numpy()
    mask = (mask.transpose(1,2,0) * 255).astype(np.uint8)
    mask2 = cv2.erode(mask, np.ones((20,20)), borderType=cv2.BORDER_CONSTANT, borderValue=0)

    mask = np.vstack((mask1[:900,:],mask2[900:,:]))

    mask = Image.fromarray(mask)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=20)).convert('L')
    mask = torch.from_numpy(np.array(mask)/255.).cuda().unsqueeze(0)



    # mask = Image.fromarray(mask)
    # mask = mask.filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
    # mask2 = torch.from_numpy(np.array(mask)/255.).cuda().unsqueeze(0)


    # print (mask1.shape,mask2.shape)

    # mask = torch.cat((mask1[:,:950,:],mask2[:,950:,:]),dim=1)
    # print (mask.shape)

    coeff = coeff.squeeze()
    
    pro_frame = transforms.functional_tensor.perspective(pro_frame, coeff)
    pro_frame = pro_frame[ :, :crop_size[0], :crop_size[1]]        

    pro_frame_bg = torch.zeros_like(ori_frame)+ori_frame

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = pro_frame
    else:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = pro_frame


    mask = transforms.functional_tensor.perspective(mask, coeff)
    mask = mask[ :, :crop_size[0], :crop_size[1]]  
    mask_bg = torch.zeros_like(ori_frame)

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        mask_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = mask
    else:
        mask_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = mask

    # mask_bg = mask_bg.cpu().numpy()
    # mask_bg = (mask_bg.transpose(1,2,0) * 255).astype(np.uint8)
    # mask_bg = cv2.dilate(mask_bg, np.ones((30,30)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # mask_bg = Image.fromarray(mask_bg)
    # mask_bg = mask_bg.filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
    # mask_bg = torch.from_numpy(np.array(mask_bg)/255.).cuda().unsqueeze(0)

    output = ori_frame*(1-mask_bg) + pro_frame_bg*mask_bg
    return output.unsqueeze(0).float(),mask_bg#pro_frame_bg




def reproject_wi_dilate(ori_frame, pro_frame, mask, coeff, crop_size, quad_0):

    ori_frame = ori_frame.squeeze()
    pro_frame = pro_frame.squeeze()
    mask = mask.squeeze(0)


    mask = mask.cpu().numpy()
    mask = (mask.transpose(1,2,0) * 255).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((50,50)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.erode(mask, np.ones((20,20)), borderType=cv2.BORDER_CONSTANT, borderValue=0)

    mask = Image.fromarray(mask)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
    mask = torch.from_numpy(np.array(mask)/255.).cuda().unsqueeze(0)



    coeff = coeff.squeeze()
    
    pro_frame = transforms.functional_tensor.perspective(pro_frame, coeff)
    pro_frame = pro_frame[ :, :crop_size[0], :crop_size[1]]        

    pro_frame_bg = torch.zeros_like(ori_frame)+ori_frame

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = pro_frame
    else:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = pro_frame


    mask = transforms.functional_tensor.perspective(mask, coeff)
    mask = mask[ :, :crop_size[0], :crop_size[1]]  
    mask_bg = torch.zeros_like(ori_frame)

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        mask_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = mask
    else:
        mask_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = mask

    # mask_bg = mask_bg.cpu().numpy()
    # mask_bg = (mask_bg.transpose(1,2,0) * 255).astype(np.uint8)
    # mask_bg = cv2.dilate(mask_bg, np.ones((30,30)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # mask_bg = Image.fromarray(mask_bg)
    # mask_bg = mask_bg.filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
    # mask_bg = torch.from_numpy(np.array(mask_bg)/255.).cuda().unsqueeze(0)

    output = ori_frame*(1-mask_bg) + pro_frame_bg*mask_bg
    return output.unsqueeze(0).float(),mask_bg#pro_frame_bg



def reproject_wi_erode(ori_frame, pro_frame, mask, coeff, crop_size, quad_0):

    ori_frame = ori_frame.squeeze()
    pro_frame = pro_frame.squeeze()
    mask = mask.squeeze(0)

    coeff = coeff.squeeze()
    # print (coeff.shape)
    # print (crop_size, quad_0)
    pro_frame = transforms.functional_tensor.perspective(pro_frame, coeff)
    pro_frame = pro_frame[ :, :crop_size[0], :crop_size[1]]        

    pro_frame_bg = torch.zeros_like(ori_frame)+ori_frame

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = pro_frame
    else:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = pro_frame


    mask = transforms.functional_tensor.perspective(mask, coeff)
    mask = mask[ :, :crop_size[0], :crop_size[1]]  
    mask_bg = torch.zeros_like(ori_frame)

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        mask_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = mask
    else:
        mask_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = mask

    mask_bg = mask_bg.cpu().numpy()
    mask_bg = (mask_bg.transpose(1,2,0) * 255).astype(np.uint8)

    mask_bg = cv2.erode(mask_bg, np.ones((30,30)), borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # mask_bg = cv2.erode(mask_bg, np.ones((10,10)), borderType=cv2.BORDER_CONSTANT, borderValue=0)


    mask_bg = Image.fromarray(mask_bg)
    mask_bg = mask_bg.filter(ImageFilter.GaussianBlur(radius=10)).convert('L')

    # mask_bg = mask_bg.filter(ImageFilter.GaussianBlur(radius=2)).convert('L')

    mask_bg = torch.from_numpy(np.array(mask_bg)/255.).cuda().unsqueeze(0)

    output = ori_frame*(1-mask_bg) + pro_frame_bg*mask_bg
    return output.unsqueeze(0).float(),mask_bg




def reproject_wi_erode_batch(ori_frames, pro_frames, masks, coeffs, crop_sizes, quad_0s):
   
    outputs = []
    # mask_bgs = []
    # print ('xxxx',ori_frames.shape, pro_frames.shape)
    for i in range(ori_frames.shape[0]):
        ori_frame = ori_frames[i]
        pro_frame = pro_frames[i]
        mask = masks[i]#.squeeze(0)
        coeff = coeffs[i]
        crop_size = [crop_sizes[0][i],crop_sizes[1][i]]
        quad_0 = [quad_0s[0][i],quad_0s[1][i],quad_0s[2][i],quad_0s[3][i]]
        # print (coeff.shape)
        pro_frame = transforms.functional_tensor.perspective(pro_frame, coeff)
        pro_frame = pro_frame[ :, :crop_size[0], :crop_size[1]]        

        pro_frame_bg = torch.zeros_like(ori_frame)+ori_frame

        if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
            pro_frame_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = pro_frame
        else:
            pro_frame_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = pro_frame


        mask = transforms.functional_tensor.perspective(mask, coeff)
        mask = mask[ :, :crop_size[0], :crop_size[1]]  
        mask_bg = torch.zeros_like(ori_frame)

        if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
            mask_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = mask
        else:
            mask_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = mask

        mask_bg = mask_bg.cpu().numpy()
        mask_bg = (mask_bg.transpose(1,2,0) * 255).astype(np.uint8)

        mask_bg = cv2.erode(mask_bg, np.ones((50,50)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        #mask_bg = cv2.dilate(mask_bg, np.ones((50,50)), borderType=cv2.BORDER_CONSTANT, borderValue=0)

        mask_bg = Image.fromarray(mask_bg)
        mask_bg = mask_bg.filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
        mask_bg = torch.from_numpy(np.array(mask_bg)/255.).cuda().unsqueeze(0)

        output = ori_frame*(1-mask_bg) + pro_frame_bg*mask_bg
        outputs.append(output)
    return torch.stack(outputs).float(),mask_bg


def reproject_wi_pb(ori_frame, pro_frame, mask, coeff, crop_size, quad_0):

    ori_frame = ori_frame.squeeze()
    pro_frame = pro_frame.squeeze()
    mask = mask.squeeze(0)

    coeff = coeff.squeeze()
    
    pro_frame = transforms.functional_tensor.perspective(pro_frame, coeff)
    pro_frame = pro_frame[ :, :crop_size[0], :crop_size[1]]        

    pro_frame_bg = torch.zeros_like(ori_frame)#+ori_frame

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = pro_frame
    else:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = pro_frame


    mask = transforms.functional_tensor.perspective(mask, coeff)
    mask = mask[ :, :crop_size[0], :crop_size[1]]  
    mask_bg = torch.zeros_like(ori_frame)

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        mask_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = mask
    else:
        mask_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = mask

    mask_bg = mask_bg.cpu().numpy()
    mask_bg = (mask_bg.transpose(1,2,0) * 255).astype(np.uint8)

    mask_bg = cv2.dilate(mask_bg, np.ones((30,30)), iterations=5)
    ori_frame = ori_frame.detach().cpu().numpy().transpose(1,2,0)
    pro_frame_bg =pro_frame_bg.detach().cpu().numpy().transpose(1,2,0)

    ori_frame = ((ori_frame+1)/2.0 * 255).astype(np.uint8)
    pro_frame_bg = ((pro_frame_bg+1)/2.0 * 255).astype(np.uint8)

    br = cv2.boundingRect(cv2.split(mask_bg)[0]) # bounding rect (x,y,width,height)
    center = (br[0] + br[2] // 2, br[1] + br[3] // 2)

    output = cv2.seamlessClone(pro_frame_bg, ori_frame, mask_bg, center, cv2.NORMAL_CLONE)

    output = (torch.from_numpy(output.transpose(2,0,1))/255.0) *2 -1


    return output.unsqueeze(0).float(),mask_bg#/255.0[:,:,0]



def reproject_wi_erode_lt(ori_frame, pro_frame, mask, coeff, crop_size, quad_0):

    ori_frame = ori_frame.squeeze()
    pro_frame = pro_frame.squeeze()
    mask_bg = mask#.squeeze(0)

    coeff = coeff.squeeze()
    
    pro_frame = transforms.functional_tensor.perspective(pro_frame, coeff)
    pro_frame = pro_frame[ :, :crop_size[0], :crop_size[1]]        

    pro_frame_bg = torch.zeros_like(ori_frame)+ori_frame

    if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = pro_frame
    else:
        pro_frame_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = pro_frame


    # mask = transforms.functional_tensor.perspective(mask, coeff)
    # mask = mask[ :, :crop_size[0], :crop_size[1]]  
    # mask_bg = torch.zeros_like(ori_frame)

    # if (int(quad_0[3]) - int(quad_0[1])) > 1024 or (int(quad_0[2]) - int(quad_0[0])) > 1024:
    #     mask_bg[:,int(quad_0[1]):int(quad_0[1]+pro_frame.shape[1]),int(quad_0[0]) : int(quad_0[0]+pro_frame.shape[2])] = mask
    # else:
    #     mask_bg[:,int(quad_0[1]):int(quad_0[3]),int(quad_0[0]) : int(quad_0[2])] = mask

    mask_bg = mask_bg.cpu().numpy()
    # print (mask_bg.shape,mask_bg.max())
    mask_bg = (mask_bg.transpose(1,2,0)).astype(np.uint8)

    mask_bg = cv2.erode(mask_bg, np.ones((30,30)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    #mask_bg = cv2.dilate(mask_bg, np.ones((50,50)), borderType=cv2.BORDER_CONSTANT, borderValue=0)

    mask_bg = Image.fromarray(mask_bg)
    mask_bg = mask_bg.filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
    mask_bg = torch.from_numpy(np.array(mask_bg)/255.).cuda().unsqueeze(0)

    output = ori_frame*(1-mask_bg) + pro_frame_bg*mask_bg
    return output.unsqueeze(0).float(),mask_bg



def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def warp_interpolate(warp_12,warp_32,mask):
    frame2_warp = (0.5 * mask * warp_12 + 0.5 * (1-mask) * warp_32) / (0.5 * mask + 0.5 * (1-mask))
    return frame2_warp


def cv2_to_pil(open_cv_image):
    return Image.fromarray(open_cv_image[:, :, ::-1].copy())

def pil_to_cv2_exp(pil_image):
    open_cv_image = np.expand_dims(np.array(pil_image),axis=0)
    return open_cv_image[:, :, ::-1].copy()

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    return open_cv_image[:, :, ::-1].copy()


def numpy_to_cv2(numpy_image):
    #open_cv_image = np.array(pil_image) 
    return numpy_image[:, :, ::-1].copy()

def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False,).cuda()
    gridY = torch.tensor(gridY, requires_grad=False,).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut




def fuse_mask(mask1,mask2):
    mask_fuse = mask1 + mask2
    mask_fuse[mask_fuse>1.0]=mask2.max()
    return mask_fuse.long()




def pre_process_video(tensor,coors,resolution=512):
    x = coors[0]
    y = coors[1]
    # gap = int(resolution/2)
    gap = 512


    if x-gap <= 0:
        x_c = 0

    else:
        x_c = x-gap
    
    if y-gap <= 0:
        y_c = 0
    else:
        y_c = y-gap
           
    tensor_crop = tensor[:,:,:,x_c:x+gap,y_c:y+gap]
    shape_s = tensor_crop.shape
    shape_t = [shape_s[0],shape_s[1],shape_s[2],resolution,resolution]
    try:
        tensor_resize = F.interpolate(tensor_crop,size=(3, resolution,resolution))
    except:
        tensor_resize = F.interpolate(tensor,size=(3, resolution,resolution))
        

    return tensor_resize

    
def pre_process(tensor,coors,resolution=640):
    x = coors[0]
    y = coors[1]
    gap = int(resolution/2)

    if x-gap <= 0:
        x_c = 0

    else:
        x_c = x-gap
    
    if y-gap <= 0:
        y_c = 0
    else:
        y_c = y-gap
           
    tensor_crop = tensor[:,:,x_c:x+gap,y_c:y+gap]

    tensor_out = transforms.Resize((resolution,resolution))(tensor_crop)
    return tensor_out

def pre_process_batch(tensors,coorss,resolution=640):
    tensor_outs = []
    # print (tensors.shape)
    # print (coorss)
    for i in range(tensors.shape[0]):
        tensor = tensors[i].unsqueeze(0)
        coors = [coorss[0][i],coorss[1][i]]
        x = coors[0]
        y = coors[1]
        gap = int(resolution/2)

        if x-gap <= 0:
            x_c = 0

        else:
            x_c = x-gap
        
        if y-gap <= 0:
            y_c = 0
        else:
            y_c = y-gap
            
        tensor_crop = tensor[:,:,x_c:x+gap,y_c:y+gap]

        tensor_out = transforms.Resize((resolution,resolution))(tensor_crop).squeeze()
        # print (tensor_out.shape)
        tensor_outs.append(tensor_out)
    return torch.stack(tensor_outs)


def slomo_interpolate(i0, i1, flow, interp, back_warp,trans_forward,trans_backward):

    i0 = trans_forward((i0+1.0)/2.0)# trans_forward(frame)
    i1 = trans_forward((i1+1.0)/2.0)

    ix = torch.cat([i0, i1], dim=1)

    flow_out = flow(ix)
    f01 = flow_out[:, :2, :, :]
    f10 = flow_out[:, 2:, :, :]


    t = 0.5
    temp = -t * (1 - t)
    co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

    ft0 = co_eff[0] * f01 + co_eff[1] * f10
    ft1 = co_eff[2] * f01 + co_eff[3] * f10

    gi0ft0 = back_warp(i0, ft0)
    gi1ft1 = back_warp(i1, ft1)

    iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
    io = interp(iy)

    ft0f = io[:, :2, :, :] + ft0
    ft1f = io[:, 2:4, :, :] + ft1
    vt0 = F.sigmoid(io[:, 4:5, :, :])
    vt1 = 1 - vt0

    gi0ft0f = back_warp(i0, ft0f)
    gi1ft1f = back_warp(i1, ft1f)

    co_eff = [1 - t, t]

    ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
           (co_eff[0] * vt0 + co_eff[1] * vt1)

    ft_p = trans_backward(ft_p)*2.0-1.0

    return ft_p


def set_requires_grad(nets, requires_grad=False):

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad






class VideoPool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

        


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)



def create_masks(border_pixels, mask, inner_dilation=0, outer_dilation=0, whole_image_border=False):
    image_size = mask.shape[2]
    grid = torch.cartesian_prod(torch.arange(image_size), torch.arange(image_size)).view(image_size, image_size,
                                                                                         2).cuda()
    image_border_mask = logical_or_reduce(
        grid[:, :, 0] < border_pixels,
        grid[:, :, 1] < border_pixels,
        grid[:, :, 0] >= image_size - border_pixels,
        grid[:, :, 1] >= image_size - border_pixels
    )[None, None].expand_as(mask)

    temp = mask
    if inner_dilation != 0:
        temp = dilation(temp, torch.ones(2 * inner_dilation + 1, 2 * inner_dilation + 1, device=mask.device),
                        engine='convolution')

    border_mask = torch.min(image_border_mask, temp)
    full_mask = dilation(temp, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=mask.device),
                         engine='convolution')
    if whole_image_border:
        border_mask_2 = 1 - temp
    else:
        border_mask_2 = full_mask - temp
    border_mask = torch.maximum(border_mask, border_mask_2)

    border_mask = border_mask.clip(0, 1)
    content_mask = (mask - border_mask).clip(0, 1)
    return content_mask, border_mask, full_mask


# 0: 'background' 1: 'skin'   2: 'nose'
# 3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
# 6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
# 9: 'r_ear'  10: 'mouth' 11: 'u_lip'
# 12: 'l_lip' 13: 'hair'  14: 'hat'
# 15: 'ear_r' 16: 'neck_l'    17: 'neck'
# 18: 'cloth'     


def calc_masks(inversion, segmentation_model, border_pixels, inner_mask_dilation, outer_mask_dilation,
               whole_image_border):
    # background_classes = [0, 18, 16]
    background_classes = [0, 18, 16, 17]


    # background_classes = [0, 18, 16, 13, 14, 17]

    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0].argmax(dim=1, keepdim=True)
    is_foreground = logical_and_reduce(*[segmentation != cls for cls in background_classes])
    foreground_mask = is_foreground.float()
    content_mask, border_mask, full_mask = create_masks(border_pixels // 2, foreground_mask, inner_mask_dilation // 2,
                                                        outer_mask_dilation // 2, whole_image_border)
    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=True)
    border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=True)
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=True)
    return content_mask, border_mask, full_mask

def calc_mask(inversion, segmentation_model):

    background_classes = [0, 18, 16, 17]
    # background_classes = [0, 18, 16, 13, 14, 17]

    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0].argmax(dim=1, keepdim=True)
    #print ('xxx',segmentation.shape)
    is_foreground = logical_and_reduce(*[segmentation != cls for cls in background_classes])
    foreground_mask = is_foreground.float()
    foreground_mask = F.interpolate(foreground_mask, (1024, 1024), mode='bilinear', align_corners=True)
    return foreground_mask





def calc_map(inversion, segmentation_model):

    # background_classes = [0, 18, 16]
    background_classes = [0, 18, 16, 17]


    # background_classes = [0, 18, 16, 13, 14, 17]

    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0]#.argmax(dim=1, keepdim=True)
    #print (segmentation.shape,segmentation[0,:,0,0].sum())
    
    segmentation = nn.Softmax(dim=1)(segmentation)
    saliency = torch.zeros_like(segmentation)[:,:2]
    #print (segmentation.shape,saliency.shape)

    for cls in range(segmentation.shape[1]):
        if cls in background_classes:
            saliency[:,0] = saliency[:,0] + segmentation[:,cls]
        else:
            saliency[:,1] = saliency[:,1] + segmentation[:,cls]
    #print (segmentation.sum(),saliency.sum())

    saliency = saliency.float()
    saliency = F.interpolate(saliency, (1024, 1024), mode='bilinear', align_corners=True)
    return saliency




def create_video(image_folder, fps=24, video_format='.mp4', resize_ratio=1):
    video_name = image_folder + video_format
    video_name = video_name.replace('frames','videos')
    video_name_c = video_name.replace('videos','videos_converted')

    # print (video_name)

    img_list = glob.glob1(image_folder,'frame*')
    img_list.sort()
    frame = cv2.imread(os.path.join(image_folder, img_list[0]))
    frame = cv2.resize(frame, (0,0), fx=resize_ratio, fy=resize_ratio) 
    height, width, layers = frame.shape
    if video_format == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif video_format == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
    for image_name in img_list:
        frame = cv2.imread(os.path.join(image_folder, image_name))
        frame = cv2.resize(frame, (0,0), fx=resize_ratio, fy=resize_ratio) 
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()
    print ('ffmpeg -i ' + video_name + ' -vcodec libx264  -f mp4 '+ video_name_c)
    os.system('ffmpeg -i ' + video_name + ' -vcodec libx264  -f mp4 '+ video_name_c)



def create_video_cat(image_folder1,image_folder2,image_folder3, fps=24, video_format='.mp4', resize_ratio=1):
    video_name = image_folder3 +  '_Compare' +video_format
    video_name = video_name.replace('frames','videos')
    print (video_name)
    video_name_c = video_name.replace('videos','videos_converted')

    img_list = glob.glob1(image_folder1,'frame*')
    img_list.sort()



    # frame = cv2.imread(os.path.join(image_folder1, img_list[0]))
    # frame = cv2.resize(frame, (0,0), fx=resize_ratio, fy=resize_ratio) 
    # height, width, layers = frame.shape


    frame1 = cv2.imread(os.path.join(image_folder1, img_list[0]))
    frame1 = cv2.resize(frame1, (0,0), fx=resize_ratio, fy=resize_ratio) 

    frame2 = cv2.imread(os.path.join(image_folder2, img_list[0]))
    frame2 = cv2.resize(frame2, (0,0), fx=resize_ratio, fy=resize_ratio) 

    frame3 = cv2.imread(os.path.join(image_folder3, img_list[0]))
    frame3 = cv2.resize(frame3, (0,0), fx=resize_ratio, fy=resize_ratio) 


    frame = concat_images_horizontally(cv2_to_pil(frame1), cv2_to_pil(frame2), cv2_to_pil(frame3))
    frame = add_texts_to_image_vertical(['original', 'inversion', 'edited'], frame)
    frame = pil_to_cv2(frame)

    print (frame.shape)
    height, width, layers = frame.shape

    if video_format == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif video_format == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter(video_name, fourcc, fps, (width*3,height))
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

    for image_name in img_list:

        frame1 = cv2.imread(os.path.join(image_folder1, image_name))
        frame1 = cv2.resize(frame1, (0,0), fx=resize_ratio, fy=resize_ratio) 

        frame2 = cv2.imread(os.path.join(image_folder2, image_name))
        frame2 = cv2.resize(frame2, (0,0), fx=resize_ratio, fy=resize_ratio) 

        frame3 = cv2.imread(os.path.join(image_folder3, image_name))
        frame3 = cv2.resize(frame3, (0,0), fx=resize_ratio, fy=resize_ratio) 


        frame = concat_images_horizontally(cv2_to_pil(frame1), cv2_to_pil(frame2), cv2_to_pil(frame3))
        frame = add_texts_to_image_vertical(['original', 'inversion', 'edited'], frame)
        frame = pil_to_cv2(frame)

        video.write(frame)
    #cap.release()
    video.release()
    cv2.destroyAllWindows()

    #ffmpeg -i /nvme/xyy/VIDEO/RIGID/exp/RAVDESS/FULL_Batch_32/test/videos/-2KGPYEFnsU_8_ori.mp4  -vcodec libx264 -f mp4 output.mp4
    print ('ffmpeg -i ' + video_name + ' -vcodec libx264  -f mp4 '+ video_name_c)
    #subprocess.run('ffmpeg -i ' + video_name + ' -vcodec libx264  -f mp4 '+ video_name_c, shell=True)
    os.system('ffmpeg -i ' + video_name + ' -vcodec libx264  -f mp4 '+ video_name_c)
    #time.sleep(10)







def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        #print (x.shape)
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def encode_segmentation_rgb(segmentation, no_neck=True):
    parse = segmentation[:,:,0]

    face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse==mouth_id)
    mouth_map[valid_index] = 255
    valid_index = np.where(parse==hair_id)
    hair_map[valid_index] = 255

    return np.stack([face_map, mouth_map, hair_map], axis=2)

    
def flip_video(x):
    num = random.randint(0, 1)
    if num == 0:
        return torch.flip(x, [2])
    else:
        return x


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)
            
    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22]).cuda()
for param in vgg16_conv_4_3.parameters():
        param.requires_grad = False

# loss function
def interp_loss(output, IT):
    It_warp = output

    recnLoss = F.l1_loss(It_warp, IT)
    prcpLoss = 0.1 * F.mse_loss(vgg16_conv_4_3(It_warp), vgg16_conv_4_3(IT))

    loss = recnLoss + prcpLoss

    return loss#, recnLoss,  prcpLoss,


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

MISSING_VALUE = -1
def cords_to_map(_cords, img_size, sigma=6):
    results = []
    for j in range(_cords.shape[0]):
        cords = _cords[j]
        result = torch.zeros(img_size + cords.shape[0:1], dtype=torch.uint8)
        for i, point in enumerate(cords):
            if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                continue
            xx, yy = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
            x = torch.exp(-((yy - int(point[0])) ** 2 + (xx - int(point[1])) ** 2) / (2 * sigma ** 2))
            result[..., i] = x
        results.append(result)
    return torch.stack(results,dim=0)



def cords_to_map_np(_cords, img_size, sigma=6):
    results = []
    for j in range(_cords.shape[0]):
        cords = _cords[j]

        result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
        for i, point in enumerate(cords):
            if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            x = np.exp(-((yy - int(point[0])) ** 2 + (xx - int(point[1])) ** 2) / (2 * sigma ** 2))
            result[..., i] = x
        results.append(result)
    return np.array(results)


def set_requires_grad(nets, requires_grad=False):

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_logistic_loss_2(real_pred, fake_pred1, fake_pred2):
    real_loss = F.softplus(-real_pred)
    fake_loss1 = F.softplus(fake_pred1)
    fake_loss2 = F.softplus(fake_pred2)

    return real_loss.mean() + fake_loss1.mean() + fake_loss2.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def discriminator_r1_loss(real_pred, real_w):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_w, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty