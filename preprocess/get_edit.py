import argparse
import math
import random
import os,time
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils
from tqdm import tqdm


import itertools


from models.encoders.psp_encoders import *
from models.stylegan2.model import *
from models.nets import *

import cv2
import glob
import numpy as np
import os
import face_alignment

from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage import io




def edit_img(edited_path, latent_path, direction):

    os.makedirs(edited_path, exist_ok=True)
    
    
    device = 'cuda'
    size = 1024
    latent = 512
    n_mlp = 8
 

    generator = Generator(size,latent,n_mlp).to(device)

    direction = direction.to(device)

    weight_ckpt = torch.load('../weight_files/stylegan2-ffhq-config-f.pt', map_location=torch.device('cpu'))["g_ema"]
    weight_ckpt = {k: v for k, v in weight_ckpt.items() if "noises" not in k}
    generator.load_state_dict(weight_ckpt,strict=False)


    latent_avg = torch.load("../weight_files/e4e_ffhq_encode.pt", map_location=torch.device('cpu'))['latent_avg'].to(device)

    code_list = glob.glob1(latent_path, 'frame*')
    code_list.sort()

    for idx, code_name in enumerate(code_list):

        if os.path.isfile(edited_path + code_name)==False:
            code_path = os.path.join(latent_path, code_name)
            code = np.load(code_path)
            #print (code.shape)
            code = torch.from_numpy(code).to(device)


            w_code = code + latent_avg.repeat(code.shape[0], 1, 1) - 3 * direction
            img_edit, _ = generator([w_code], input_is_latent=True, randomize_noise=False)

            utils.save_image(
                img_edit,
                edited_path + code_name.replace('npy','jpg'),
                normalize=True,
                range=(-1, 1),
            )


path = 'RAVDESS/'

direction = torch.load('./directions/interfacegan_directions/age.pt')

for video in os.listdir(path+'video/'):
    print (video)

    latent_path = path+'frame_aligned_latent/'+video.split(".")[0]+"/"
    edited_path = path+'frame_aligned_edit_age/'+video.split(".")[0]+"/"

    edit_img(edited_path, latent_path, direction)



