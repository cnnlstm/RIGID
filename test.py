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

from datasets import *
from torch.autograd import Variable
import matplotlib as mlb


import itertools
from tensorboardX import SummaryWriter
from utils import *
from util import *



from models.encoders.psp_encoders import *
from models.stylegan2.model import *
from models.nets import *
from models.lstm_model import *
from models.seg_model import *

import networks

from criteria.id_loss import IDLoss
from criteria.lpips.lpips import LPIPS


try:
    from networks.resample2d_package.resample2d import Resample2d
except:
    from .networks.resample2d_package.resample2d import Resample2d


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)





def fuse_mask(mask1,mask2):
    mask_fuse = mask1 + mask2
    mask_fuse[mask_fuse>1.0]=mask2.max()
    return mask_fuse.long()



def create_video_cat2(image_folder1,image_folder2, attr, fps=24, video_format='.mp4', resize_ratio=1):
    video_name = image_folder2 +  '_Compare' +video_format
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

    # frame3 = cv2.imread(os.path.join(image_folder3, img_list[0]))
    # frame3 = cv2.resize(frame3, (0,0), fx=resize_ratio, fy=resize_ratio) 


    frame = concat_images_horizontally(cv2_to_pil(frame1), cv2_to_pil(frame2))
    frame = add_texts_to_image_vertical(['Original', attr], frame)
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

        # frame3 = cv2.imread(os.path.join(image_folder3, image_name))
        # frame3 = cv2.resize(frame3, (0,0), fx=resize_ratio, fy=resize_ratio) 


        frame = concat_images_horizontally(cv2_to_pil(frame1), cv2_to_pil(frame2))
        frame = add_texts_to_image_vertical(['Original', attr], frame)
        frame = pil_to_cv2(frame)

        video.write(frame)
    #cap.release()
    video.release()
    cv2.destroyAllWindows()

    #ffmpeg -i /nvme/xyy/VIDEO/RIGID/exp/RAVDESS/FULL_Batch_32/test/videos/-2KGPYEFnsU_8_ori.mp4  -vcodec libx264 -f mp4 output.mp4
    print ('ffmpeg -i ' + video_name + ' -vcodec libx264  -f mp4 '+ video_name_c)
    #subprocess.run('ffmpeg -i ' + video_name + ' -vcodec libx264  -f mp4 '+ video_name_c, shell=True)
    os.system('ffmpeg -i ' + video_name + ' -vcodec libx264  -f mp4 '+ video_name_c)

    os.system('rm -rf ' + video_name)

    #time.sleep(10)



def train(args, train_loader, encoder, generator, segmodel, edit_model,noises_base, device):



    generator.eval()
    segmodel.eval()
    edit_model.eval()

    trans_256 = transforms.Resize(256)
    trans_1024 = transforms.Resize(1024)

    # attr_dict = {'Arched_Eyebrows': 1,  \
    #         'Bald': 4, 'Big_Lips': 6,
    #         'Bushy_Eyebrows': 12, 'Chubby': 13,  \
    #         'Eyeglasses': 15, 'Heavy_Makeup': 18,\
    #         'Male': 20, 'Narrow_Eyes': 23,  \
    #         'Smiling': 31, 'Young': 39}


    
    # attr_dict = {'Arched_Eyebrows': 1,  \
    #         'Bald': 4, 'Big_Lips': 6,
    #         'Bushy_Eyebrows': 12, 'Chubby': 13,  \
    #         'Eyeglasses': 15, 'Heavy_Makeup': 18,\
    #         'Male': 20, 'Narrow_Eyes': 23,  \
    #         'Smiling': 31, 'Young': 39}


    # attr_dict = {'Arched_Eyebrows': 1,  \
    #         'Bushy_Eyebrows': 12, 'Chubby': 13,  \
    #         'Eyeglasses': 15, 'Heavy_Makeup': 18,\
    #         'Narrow_Eyes': 23,  \
    #         'Young': 39}

    # '5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
    #             'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \

    # attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
    #         'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
    #         'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
    #         'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
    #         'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
    #         'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
    #         'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
    #         'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}


    attr_dict = {'Male': 20,}
        

    zero_latent = torch.zeros((args.batch,4,512)).to(device).detach()
    channels = {
            4: 512,
            8: 512 ,
            16: 512 ,
            32: 512 ,
            64: 512 ,
            128: 256 ,
            256: 128,
        }



    for iteration, batch in enumerate(train_loader):


            video_ori,video_256, video_1024,video_rec,video_code,video_coeff,video_center,video_crop_size,video_quad_0,video_name,video_slices = batch



            for N, attr in enumerate(attr_dict.keys()):

                        # target = video_name[0]+"_"+attr+"_rep_Compare.mp4"
                        # print (target)
                        # if target in os.listdir('video_selected/'):
                        if attr!='Pose':
                            attr_index = attr_dict[attr]
                            edit_model.load_state_dict(torch.load(f"./editings/latentT_weights/tnet_{attr_index}.pth.tar"))
                        else:
                            eigvec = torch.load('./editings/closed_form.pt')["eigvec"].to(device)
                            pose_direction = eigvec[:, 7].unsqueeze(0) * 2.5

                        # if attr=='Eyeglasses':
                        #     args.beta = -2
                        # else:
                        #     args.beta = 1
                        

                        save_path_edit = _dirs[0]+video_name[0]+"_"+attr+'_aligned/'
                        os.makedirs(save_path_edit, exist_ok=True)

                        save_path_edit_rep = _dirs[0]+video_name[0]+"_"+attr+"_rep"
                        os.makedirs(save_path_edit_rep, exist_ok=True)

                        save_path_rec = _dirs[0]+video_name[0]+"_rec_aligned"
                        os.makedirs(save_path_rec, exist_ok=True)

                        save_path_rec_rep = _dirs[0]+video_name[0]+"_rec_rep"
                        os.makedirs(save_path_rec_rep, exist_ok=True)


                        save_path_ori = _dirs[0]+video_name[0]+"_ori"
                        os.makedirs(save_path_ori, exist_ok=True)


                        save_path_ori_aligned = _dirs[0]+video_name[0]+"_ori_aligned"
                        os.makedirs(save_path_ori_aligned, exist_ok=True)






                        lstm_state = None


                        video_ori256 = []
                        video_inv = []
                        video_edit = []

                        t0 = 0
                        t1 = 0



                        for t in range(1, len(video_ori)):
                            
                            frame_ori1 = video_ori[t-1].to(device)
                            frame_i1 = video_256[t-1].to(device)
                            frame_rec1 = video_rec[t-1].to(device)
                            coeff_1 = video_coeff[t-1].to(device)
                            crop_size_1 = video_crop_size[t-1]
                            quad_1 = video_quad_0[t-1]
                            code_i1 = video_code[t-1].to(device)
                            frame_center1 = video_center[t-1]



                            frame_ori2 = video_ori[t].to(device)
                            frame_i2= video_256[t].to(device)
                            frame_rec2 = video_rec[t].to(device)
                            coeff_2 = video_coeff[t].to(device)
                            crop_size_2 = video_crop_size[t]
                            quad_2 = video_quad_0[t]
                            code_i2 = video_code[t].to(device)
                            frame_center2 = video_center[t]


                            mask_i2 = calc_mask(frame_i2, segmodel).detach()

                            if t == 1:

                                mask_i1 = calc_mask(frame_i1, segmodel).detach()
                                frame_o1 = frame_i1
                                frame_ori_start = frame_ori1
                                frame_rep_start,_ = reproject_wi_erode(frame_ori1, trans_1024(frame_i1), mask_i1, coeff_1, crop_size_1, quad_1)
                                frame_rep1 = frame_rep_start

                                if attr!='Pose':
                                    code_e1 = edit_model(code_i1.view(code_i1.size(0), -1), args.beta).view(code_i1.size())
                                else:
                                    code_e1 = code_i1 + pose_direction



                                # frame_o2, _ = generator([code], noise=noises_base, input_is_latent=True, randomize_noise=False)

                                # frame_e2, _ = generator([code_edit], noise=noises_base, input_is_latent=True, randomize_noise=False)




                                frame_e1, _ = generator([code_e1.squeeze(1)], noise=noises_base, input_is_latent=True, randomize_noise=False)
                                frame_e1 = frame_e1.detach() 
                                mask_e1 = calc_mask(frame_e1, segmodel).detach()
                                mask_fuse1 = fuse_mask(mask_i1,mask_e1).detach()
                                frame_rep1_e,_ = reproject_wi_erode(frame_ori1, trans_1024(frame_e1), mask_fuse1, coeff_1, crop_size_1, quad_1)

                                base_code = code_i1


                            else:

                                frame_o1 = frame_o2.detach()   
                                frame_o1.requires_grad = False 

                                frame_e1 = frame_e2.detach()   
                                frame_e1.requires_grad = False 

                                frame_rep1 = frame_rep2.detach()
                                frame_rep1.requires_grad = False 



                            code_i2 = torch.cat([code_i2[:,:10],base_code[:,10:]],dim=1)

                            inputs = torch.cat((frame_i1, frame_i2, frame_o1, frame_e1), dim=1)

                            code_offset, lstm_state, noises = encoder(inputs, lstm_state, args.noise_res)

                            for i,j in enumerate(noises_base):
                                if j.shape[-1] == noises.shape[-1]:
                                    noises_base[i] = noises
                                else:
                                    noises_base[i] =  noises_base[i].to(device)

                            lstm_state = repackage_hidden(lstm_state)


                            code_offset = torch.cat([code_offset,zero_latent],dim=1)
                            code = code_i2 + code_offset



                            if attr!='Pose':
                                code_edit = edit_model(code.view(code.size(0), -1), 1).view(code.size())
                            else:
                                code_edit = code + pose_direction


                            code_edit = torch.cat([code_edit[:,:10],code_e1[:,10:]],dim=1) # edit code use the layer swap with 1st edit code.
                            code_cat = torch.cat([code,code_edit],dim=0)
                            frames, _ = generator([code_cat], noise=noises_base, input_is_latent=True, randomize_noise=False)



                            frame_o2 = frames[0].unsqueeze(0)
                            frame_e2 = frames[1].unsqueeze(0)

                            mask_e2 = calc_mask(frame_e2, segmodel).detach()
                            mask_fuse2 = fuse_mask(mask_i2, mask_e2).detach()


                            frame_rep2_edit,_ = reproject_wi_erode(frame_ori2,  trans_1024(frame_e2), mask_fuse2,  coeff_2, crop_size_2, quad_2)
                            frame_rep2,_ = reproject_wi_erode(frame_ori2,  trans_1024(frame_o2), mask_i2, coeff_2, crop_size_2, quad_2)
                            frame_rep2_ori256,mask_bg = reproject_wi_erode(frame_ori2,  trans_1024(frame_i2), mask_i2, coeff_2, crop_size_2, quad_2)


                            utils.save_image(
                                    frame_rep2_ori256.detach().cpu(),
                                    save_path_ori+'/{:04d}.jpg'.format(t-1),
                                    nrow=1,
                                    normalize=True,
                                    range=(-1, 1),
                                )

                            utils.save_image(
                                    frame_rep2.detach().cpu(),
                                    save_path_rec_rep+'/{:04d}.jpg'.format(t-1),
                                    nrow=1,
                                    normalize=True,
                                    range=(-1, 1),
                                )
                            utils.save_image(
                                    frame_e2.detach().cpu(),
                                    save_path_edit+'/{:04d}.jpg'.format(t-1),
                                    nrow=1,
                                    normalize=True,
                                    range=(-1, 1),
                                )
                            utils.save_image(
                                    frame_rep2_edit.detach().cpu(),
                                    save_path_edit_rep+'/{:04d}.jpg'.format(t-1),
                                    nrow=1,
                                    normalize=True,
                                    range=(-1, 1),
                                )
                            utils.save_image(
                                    frame_o2.detach().cpu(),
                                    save_path_rec+'/{:04d}.jpg'.format(t-1),
                                    nrow=1,
                                    normalize=True,
                                    range=(-1, 1),
                                )

                            utils.save_image(
                                    frame_i2.detach().cpu(),
                                    save_path_ori_aligned+'/{:04d}.jpg'.format(t-1),
                                    nrow=1,
                                    normalize=True,
                                    range=(-1, 1),
                                )











if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--e_ckpt", type=str, default=None)


    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--iter", type=int, default=500001)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--lpips", type=float, default=0.8)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--st", type=float, default=50.0)
    parser.add_argument("--lt", type=float, default=50.0)


    parser.add_argument("--beta", type=float, default=0.75)


    parser.add_argument("--noise_res", type=int, default=32)


    parser.add_argument("--id", type=float, default=0.1)
    parser.add_argument("--adv", type=float, default=0.5) 
    parser.add_argument("--modal", type=str, default='inversion')  
    parser.add_argument("--noise", action="store_true",default=True)  



    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--tensorboard", action="store_true",default=True)
    

    args = parser.parse_args()


    device = args.device

    args.start_iter = 0

    args.list = './text/CELEBV-HQ-800-Test.txt'
    # args.list = '/mnt/petrelfs/xuyangyang.p/RIGID/text/CELEBV-HQ-800-Test-Seen.txt'


    args.num_continuous = 10


    args.size = 256
    args.latent = 512
    args.n_mlp = 8
    args.channel_multiplier = 2


    args.rgb_max = 1.0
    args.fp16 = False
    args.frame_skip = 1



    encoder = GPEN_LSTM_Noise_Encoder_Selected(input_channel=12,size=args.size).to(device)
    generator = Generator(args.size,args.latent,args.n_mlp).to(device)



    _dirs = ['./exp/edit/']

    for x in _dirs:
        try:
            os.makedirs(x)
        except:
            pass

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    to_tensor_256 = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    to_tensor_1024 = transforms.Compose([
        transforms.Resize(1024),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    dataset = Video_Continuous(args.list,num_continuous=args.num_continuous,modal='test',transform=to_tensor,transform_256=to_tensor_256,transform_1024=to_tensor_1024)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False)



    segmodel = BiSeNet(19).eval().to(device).requires_grad_(False)
    segmodel.load_state_dict(torch.load("./weight_files/79999_iter.pth",map_location=torch.device('cpu')))
    edit_model = F_mapping(mapping_lrmul= 1, mapping_layers=14, mapping_fmaps=512, mapping_nonlinearity = 'linear').to(device)




    e_ckpt = torch.load(args.e_ckpt,  map_location=torch.device('cpu'))
    encoder = torch.nn.DataParallel(encoder)
    generator = torch.nn.DataParallel(generator)

    encoder.load_state_dict(e_ckpt["e"])
    generator.load_state_dict(e_ckpt["g"])

    noises_base = e_ckpt["n"]



    train(args, train_loader, encoder, generator, segmodel, edit_model, noises_base, device)


