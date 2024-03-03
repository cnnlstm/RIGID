#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch,shutil
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

### custom lib
import networks
import util



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='optical flow estimation')

    ### testing options
    parser.add_argument('-model',           type=str,     default="FlowNet2",   help='Flow model name')



    opts = parser.parse_args()


    ### FlowNet options
    opts.rgb_max = 1.0
    opts.fp16 = False

    print(opts)


    # ### initialize FlowNet
    # model = networks.__dict__[opts.model](opts)

    # ### load pre-trained FlowNet
    # model_filename = os.path.join("pretrained_models", "%s_checkpoint.pth.tar" %opts.model)
    # print("===> Load %s" %model_filename)
    # checkpoint = torch.load(model_filename)
    # model.load_state_dict(checkpoint['state_dict'])



    # flow_warping = Resample2d().cuda()
    model = networks.FlowNet2(opts)
    checkpoint = torch.load("./weight_files/FlowNet2_checkpoint.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()


    model.eval()

    ### load image list
    # list_filename = './text/CELEBV-HQ-800-Test-Seen.txt'
    list_filename = './text/CELEBV-HQ-800-Test-Unseen.txt'
    
    # list_filename = './text/CELEBV-HQ-Selected-Test.txt'

    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

   
    for video in video_list:
        video = video.split(" ")[0]

        frame_dir = '/mnt/petrelfs/xuyangyang.p/RIGID/comparison/inversion/GT/'+video.split('/')[-2]
        print (frame_dir)
        
        # fw_flow_dir = video.replace('frame_aligned','fw_flow')
        # if not os.path.isdir(fw_flow_dir):
        #     os.makedirs(fw_flow_dir)

        fw_occ_dir = frame_dir.replace('GT','fw_occlusion2')
        if not os.path.isdir(fw_occ_dir):
            os.makedirs(fw_occ_dir)
        print (fw_occ_dir)
        # fw_rgb_dir =  video.replace('frame_aligned','fw_flow_rgb')
        # if not os.path.isdir(fw_rgb_dir):
        #     os.makedirs(fw_rgb_dir)

        # shutil.rmtree(frame_dir, ignore_errors=True)
        # os.rmdir(fw_occ_dir)
        # os.rmdir(fw_flow_dir)

        frame_list = glob.glob(os.path.join(frame_dir, "*jpg"))
        # print (frame_list)
        for t in range(len(frame_list) - 1):
            
            print("Compute flow on frame %d" %( t))

            ### load input images 
            img1 = util.read_img(os.path.join(frame_dir, "%04d.jpg" %(t)))
            img2 = util.read_img(os.path.join(frame_dir, "%04d.jpg" %(t + 1)))
            
            ### resize image
            size_multiplier = 64
            H_orig = img1.shape[0]
            W_orig = img1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)
            
            img1 = cv2.resize(img1, (W_sc, H_sc))
            img2 = cv2.resize(img2, (W_sc, H_sc))
        
            with torch.no_grad():

                ### convert to tensor
                img1 = util.img2tensor(img1).cuda()
                img2 = util.img2tensor(img2).cuda()
        
                ### compute fw flow
                fw_flow = model(img1, img2)
                fw_flow = util.tensor2img(fw_flow)
            
                ### compute bw flow
                bw_flow = model(img2, img1)
                bw_flow = util.tensor2img(bw_flow)


            ### resize flow
            fw_flow = util.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig) 
            bw_flow = util.resize_flow(bw_flow, W_out = W_orig, H_out = H_orig) 
            
            ### compute occlusion
            fw_occ = util.detect_occlusion(bw_flow, fw_flow)

            ### save flow
            # output_flow_filename = os.path.join(fw_flow_dir, "%04d.flo" %t)
            # if not os.path.exists(output_flow_filename):
            #     util.save_flo(fw_flow, output_flow_filename)
        
            ### save occlusion map
            output_occ_filename = os.path.join(fw_occ_dir, "%04d.png" %t)
            if not os.path.exists(output_occ_filename):
                util.save_img(fw_occ, output_occ_filename)

            ### save rgb flow
            # output_filename = os.path.join(fw_rgb_dir, "%04d.png" %t)
            # if not os.path.exists(output_filename):
            #     flow_rgb = util.flow_to_rgb(fw_flow)
            #     util.save_img(flow_rgb, output_filename)



