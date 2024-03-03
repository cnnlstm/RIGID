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

# from torch.autograd import Variable
# import matplotlib as mlb


import itertools


from models.encoders.psp_encoders import *
from models.stylegan2.model import *
# from models.nets import *

import cv2
import glob
import numpy as np
import os
import face_alignment

from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage import io


# if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--e_ckpt", type=str, default=None)

    # parser.add_argument("--image_path", type=str, default='/home/xuyangyang/CelebAMask-HQ/CelebA-HQ-img')
    
    # parser.add_argument("--device", type=str, default='cuda')
    # parser.add_argument("--iter", type=int, default=500001)
    # parser.add_argument("--batch", type=int, default=2)
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--local_rank", type=int, default=0)

    # parser.add_argument("--lpips", type=float, default=0.8)
    # parser.add_argument("--l2", type=float, default=1.0)


    # parser.add_argument("--id", type=float, default=0.1)
    # parser.add_argument("--adv", type=float, default=0.5)  


    # parser.add_argument("--r1", type=float, default=10)
    # parser.add_argument("--d_reg_every", type=int, default=16)
    # parser.add_argument("--tensorboard", action="store_true",default=True)
    

def encode_latent(aligned_path, latent_path, rec_path):

    os.makedirs(latent_path, exist_ok=True)
    os.makedirs(rec_path, exist_ok=True)
    
    

    device = 'cuda'
    size = 256
    latent = 512
    n_mlp = 8
 


    encoder_w = Encoder4Editing(50,mode = 'ir_se').to(device)
    generator = Generator(size,latent,n_mlp).to(device)


    weight_ckpt = torch.load('./weight_files/stylegan2-ffhq-256x256.pt', map_location=torch.device('cpu'))["g_ema"]
    weight_ckpt = {k: v for k, v in weight_ckpt.items() if "noises" not in k}
    generator.load_state_dict(weight_ckpt,strict=False)

    


    weight_ckpt = torch.load("./weight_files/e4e_ffhq_encode_256.pt", map_location=torch.device('cpu'))["state_dict"]
    cache = {}
    for x in weight_ckpt.keys():
        if 'encoder' in x:
            cache[x.split('encoder.')[1]]=weight_ckpt[x]
    encoder_w.load_state_dict(cache)

    latent_avg = torch.load("./weight_files/latent_avg_stylegan2_256.pt", map_location=torch.device('cpu')).to(device)


    to_tensor = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    img_list = glob.glob1(aligned_path, '*jpg')
    img_list.sort()

    for idx, img_name in enumerate(img_list):
            print (idx,img_name)
            #if os.path.isfile(rec_path + img_name)==False:
            img_path = os.path.join(aligned_path, img_name)
            img = Image.open(img_path)
            img = to_tensor(img).to(device).unsqueeze(0)

            w_code = encoder_w(img) + latent_avg.repeat(img.shape[0], 1, 1)
            img_rec, _ = generator([w_code], input_is_latent=True, randomize_noise=False)
            
            # print (w_code.detach().cpu().numpy().shape)
            np.save(latent_path+img_name.split(".")[0]+'.npy', w_code.detach().cpu().numpy())

            utils.save_image(
                img_rec,
                rec_path + img_name,
                normalize=True,
                range=(-1, 1),
            )


# path = 'CELEBV-HQ-Unseen-Test/'
# path = "/data/xyy/RIGID-A100/CELEBV-HQ-800/"

path = "Pixar/"




for i,video in enumerate(os.listdir(path+'frame/')):
    print (i,video)

    aligned_path = path+'frame_aligned/'+video.split(".")[0]+"/"
    latent_path = path+'frame_aligned_latent_256/'+video.split(".")[0]+"/"
    rec_path = path+'frame_aligned_rec_256/'+video.split(".")[0]+"/"
    encode_latent(aligned_path, latent_path, rec_path)

#     encode_latent(aligned_path, latent_path, rec_path)


# aligned_path = '../Dataset/CelebAMask-HQ/CelebA-HQ-img/'

# # for i,img in enumerate(os.listdir(path)):
# #     print (i,img)
# #     aligned_path = path+img
# latent_path =  '../Dataset/CelebAMask-HQ/CelebA-HQ-latent-e4e/'
# rec_path = '../Dataset/CelebAMask-HQ/CelebA-HQ-rec/'




