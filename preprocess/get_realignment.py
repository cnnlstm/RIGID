import argparse
import copy
import glob
import numpy as np
import os
import torch
import yaml
import time 

from PIL import Image
from torchvision import transforms, utils, models

from utils.video_utils import *

path = 'RAVDESS/'

for video in os.listdir(path+'video/'):
    print (video)

    oriframe_path = path+'frame/'+video.split(".")[0]+"/"
    aligned_path = path+'frame_aligned/'+video.split(".")[0]+"/"
    processed_path = path+'frame_aligned_rec/'+video.split(".")[0]+"/"
    reproject_path = path+'frame_aligned_rec_rep/'+video.split(".")[0]+"/"
    video_reproject(oriframe_path, processed_path, reproject_path, aligned_path, seamless=False)
    create_video(reproject_path, video_format='.avi', resize_ratio=1)

