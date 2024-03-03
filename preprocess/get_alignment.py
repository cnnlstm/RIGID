import argparse
import copy
import glob
import numpy as np
import os
import torch
import time 

from PIL import Image
from torchvision import transforms, utils, models

from utils.video_utils import *

path = 'OOD/'



for i,video in enumerate(os.listdir(path+'frame/')):
    # print (i,video)
    #video_path = path+'video/'+video
    frame_path = path+'frame/'+video.split(".")[0]+"/"
    # for frame in os.listdir(frame_path):
    #     os.rename(frame_path+frame, frame_path+'/frame%04d' %int(frame.split(".")[0])+'.jpg' )

    # for i,frame in enumerate(sorted(os.listdir(frame_path))):

    #     old = frame_path+'/'+frame
    #     new = frame_path+'/frame%04d.jpg' % i
    #     os.rename(old,new)


    aligned_path = path+'frame_aligned/'+video.split(".")[0]+"/"
    print (aligned_path)
    #if os.path.isdir(aligned_path)==False:
    align_frames(frame_path, aligned_path, output_size=1024, optical_flow=True, filter_size=3)

    #     try:
    #         align_frames(frame_path, aligned_path, output_size=1024, optical_flow=True, filter_size=3)
    #     except:
    #         pass
    # if len(glob.glob1(frame_path, 'frame*')) == len(glob.glob1(aligned_path, 'frame*')):
    #     # print (aligned_path)
    #     # print (i,video)

    #     try:
    #         align_frames(frame_path, aligned_path, output_size=1024, optical_flow=True, filter_size=3)
    #     except:
    #         print (aligned_path)


'''
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

path = '../Dataset/RAVDESS/'


for video in os.listdir(path+'video/'):
    print (video)
    #video_path = path+'video/'+video
    frame_path = path+'frame/'+video.split(".")[0]+"/"
    aligned_path = path+'frame_aligned/'+video.split(".")[0]+"/"
    if len(glob.glob1(frame_path, 'frame*')) != len(glob.glob1(aligned_path, 'frame*')):
        try:
            align_frames(frame_path, aligned_path, output_size=1024, optical_flow=True, filter_size=3)
        except:
            pass

'''