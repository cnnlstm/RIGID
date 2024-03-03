import argparse
import copy
import glob
import numpy as np
import os
import time 



# path = './CELEBV-HQ-Unseen-Test/frame_aligned/'

# txt = open('text/CELEBV-HQ-Unseen-Test.txt','a')

# for video in os.listdir(path):
#     print (video)
#     txt.write(path+video+"/ "+str(len(os.listdir(path+video)))+"\n")



path = './Pixar/frame_aligned/'

txt = open('text/Pixar.txt','a')#.readlines()
# for x in txt:
#     print (x)


for video in os.listdir(path):
    print (video)
    txt.write(path+video+"/ "+str(len(os.listdir(path+video)))+"\n")

