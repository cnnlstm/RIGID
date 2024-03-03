import argparse
import copy
import glob
import numpy as np
import os
import time 


from utils.video_utils import *

path = 'CELEBV-HQ-800/'
j = 0
for i,video in enumerate(sorted(os.listdir(path+'video/'))):
    video_path = path+'video/'+video
    frame_path = path+'frame/'+video.split(".")[0]
    # if os.path.isdir(frame_path)==False:
    #     video_to_frames(video_path, frame_path)
    if len(os.listdir(frame_path))==0:
        j = j+1
        video_to_frames(video_path, frame_path)
        print (j,video)



