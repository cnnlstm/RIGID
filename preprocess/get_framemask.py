import argparse
import copy
import glob
import numpy as np
import os
import torch
import time 

from PIL import Image
from torchvision import transforms, utils, models
from face_parsing_master.model import BiSeNet

from utils.video_utils import *




path = 'STIT-Videos/'


to_tensor = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
net.load_state_dict(torch.load('./weight_files/79999_iter.pth'))
net.eval()


with torch.no_grad():

    for video in os.listdir(path+'frame_aligned/'):
        print (video)
        frame_path = path+'frame_aligned/'+video.split(".")[0]
        
        mask_path = path+'frame_aligned_parsing/'+video.split(".")[0]
        os.makedirs(mask_path, exist_ok=True)

        frame_list = glob.glob1(frame_path, 'frame*')
        frame_list.sort()
        for idx,frame in enumerate(frame_list):

            img = Image.open(os.path.join(frame_path, frame))
            img = to_tensor(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            #parsing = transforms.Resize(1024)(out)
            # print (out.shape)
            parsing = out.squeeze(0).cpu().argmax(0)
            #print (parsing.shape)
            
            parsing = torch.stack([parsing,parsing,parsing]).unsqueeze(0).float()
            #print (parsing.shape)

            parsing = torch.nn.functional.interpolate(parsing, size=(1024,1024)).squeeze(0).numpy()
            #parsing = transforms.Resize(1024)(parsing)

            # print (parsing.shape)
            # print (parsing[0].sum())
            # print (parsing[1].sum())
            # print (parsing[2].sum())

            #.numpy()

            cv2.imwrite(mask_path + "/mask%04d.jpg"%idx, parsing[0]);




