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

from datasets2 import *
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
from models.mask_model import *
from models.discriminator import *


import networks

from criteria.lpips.lpips import LPIPS
from criteria.gan_loss import AdversarialLoss



try:
    from networks.resample2d_package.resample2d import Resample2d
except:
    from .networks.resample2d_package.resample2d import Resample2d

setup_seed(1024)


def train(args, train_loader, encoder, generator,  flownet, segmodel, edit_model, masknet, flow_warping, flowBackWarp, optimizer, noises_base, device):


    LPIPS_loss = LPIPS(net_type='alex').to(device).eval()



    flownet.eval()
    segmodel.eval()
    masknet.eval()
    edit_model.eval()


    trans_1024 = transforms.Resize(1024)

    logger = SummaryWriter(logdir=_dirs[2])


    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    

    train_loader = sample_data(train_loader)

    attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
            'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, }

    for idx in pbar:
            total_iter = idx + args.start_iter

            if total_iter > args.iter:
                print("Done!")
                break

            time0 = time.time()
            batch = next(train_loader)

            video_ori,video_256,video_rec,video_code,video_coeff,video_center,video_crop_size,video_quad_0,video_name,video_slices = batch



            lstm_state = None

            attr = random.choice(list(attr_dict.keys()))
            if attr!='Pose':
                attr_index = attr_dict[attr]
                edit_model.load_state_dict(torch.load(f"./editings/latentT_weights_paral/tnet_{attr_index}.pth.tar"))
            else:
                eigvec = torch.load('./editings/closed_form.pt')["eigvec"].to(device)
                pose_direction = eigvec[:, 7].unsqueeze(0) * 2

            

            time1 = time.time()
            
            random_n = torch.randn(args.batch*args.num_continuous,512).to(device)


            video_p = []
            video_e = []

            video_i = []
            video_o = []


            video_ori1 = []
            video_ori2 = []
            video_rep1 = []



            for t in range(1, args.num_continuous):
                
                frame_ori1 = video_ori[t-1].to(device)
                frame_i1 = video_256[t-1].to(device)
                frame_rec1 = video_rec[t-1].to(device)
                coeff_1 = video_coeff[t-1].to(device)
                crop_size_1 = video_crop_size[t-1]
                quad_1 = video_quad_0[t-1]
                code_i1 = video_code[t-1].to(device)



                frame_ori2 = video_ori[t].to(device)
                frame_i2= video_256[t].to(device)
                frame_rec2 = video_rec[t].to(device)
                coeff_2 = video_coeff[t].to(device)
                crop_size_2 = video_crop_size[t]
                quad_2 = video_quad_0[t]
                code_i2 = video_code[t].to(device)


                mask_i2 = calc_mask(frame_i2, segmodel).detach()



                if t == 1:

                    mask_i1 = calc_mask(frame_i1, segmodel).detach()
                    frame_o1 = frame_i1
                    frame_ori_start = frame_ori1

                    frame_rep_start,_ = reproject_batch(frame_ori1, trans_1024(frame_i1), mask_i1, coeff_1, crop_size_1, quad_1)

                    frame_rep1 = frame_rep_start


                    if attr!='Pose':
                        code_e1 = edit_model(code_i1.view(code_i1.size(0), -1), 1).view(code_i1.size())
                    else:
                        code_e1 = code_i1 + pose_direction

                    # print (code_e1.shape)
                    frame_e1, _ = generator(styles=[code_e1], input_is_latent=True, randomize_noise=False)
                    frame_e1 = frame_e1.detach() 
                    mask_e1 = calc_mask(frame_e1, segmodel).detach()
                    mask_fuse1 = fuse_mask(mask_i1,mask_e1).detach()

                    frame_rep1_e,_ = reproject_batch(frame_ori1, trans_1024(frame_e1), mask_fuse1, coeff_1, crop_size_1, quad_1)


                    base_code = code_i1
                    frame_basemask = video_center[t-1]


                else:

                    frame_o1 = frame_o2.detach()   
                    frame_o1.requires_grad = False 

                    frame_e1 = frame_e2.detach()   
                    frame_e1.requires_grad = False 

                    frame_rep1 = frame_rep2.detach()
                    frame_rep1.requires_grad = False 



                inputs = torch.cat((frame_i1, frame_i2, frame_o1, frame_e1), dim=1)
                code_offset, lstm_state = encoder(inputs, lstm_state)


                code_i2 = torch.cat([code_i2[:,:10],base_code[:,10:]],dim=1)
                
                zero_latent = torch.zeros((code_offset.shape[0],4,512)).to(device).detach()

                code_offset = torch.cat([code_offset,zero_latent],dim=1)


                code = code_i2 + code_offset
                lstm_state = repackage_hidden(lstm_state)

                # for i,j in enumerate(noises_base):
                #     if j.shape[-1] == noises.shape[-1]:
                #         noises_base[i] = noises
                #     else:
                #         noises_base[i] =  noises_base[i].to(device)


                # code = code_i2 + code_offset
                # print ('attention', code.shape)

                # code_lf = code[:,:10]
                # code_hf = code[:,10:]

                if attr!='Pose':
                    code_edit = edit_model(code.view(code.size(0), -1), 1).view(code.size())
                else:
                    code_edit = code + pose_direction


                frame_o2, _ = generator([code], noise=noises_base, input_is_latent=True, randomize_noise=False)

                frame_e2, _ = generator([code_edit], noise=noises_base, input_is_latent=True, randomize_noise=False)



                video_i.append(frame_i2)
                video_o.append(frame_o2)



                mask_e2 = calc_mask(frame_e2, segmodel).detach()
                mask_fuse2 = fuse_mask(mask_i2, mask_e2).detach()


                frame_rep2,_ = reproject_batch(frame_ori2,  trans_1024(frame_o2), mask_i2, coeff_2, crop_size_2, quad_2)
                frame_rep2_e,_ = reproject_batch(frame_ori2,  trans_1024(frame_e2), mask_fuse2, coeff_2, crop_size_2, quad_2)




                video_ori1.append(frame_ori1)
                video_ori2.append(frame_ori2)
                video_rep1.append(frame_rep1)



                if t == 1: # append the first frame for visualization
                    video_p.append(frame_rep1.detach().cpu())
                    video_e.append(frame_rep1_e)
                video_p.append(frame_rep2.detach().cpu())
                video_e.append(frame_rep2_e)


            video_i =  torch.stack(video_i).reshape(-1,3,256,256)
            video_o =  torch.stack(video_o).reshape(-1,3,256,256)


            mse_loss = F.mse_loss(video_o, video_i) * args.l2
            lpips_loss = LPIPS_loss(video_o, video_i) * args.lpips



            video_ori1 = torch.stack([pre_process_batch(video_ori1[i],frame_basemask) for i in range(len(video_ori1))]).reshape(-1,3,640,640)
            video_ori2 = torch.stack([pre_process_batch(video_ori2[i],frame_basemask) for i in range(len(video_ori2))]).reshape(-1,3,640,640)
            video_rep1 = torch.stack([pre_process_batch(video_rep1[i],frame_basemask) for i in range(len(video_rep1))]).reshape(-1,3,640,640)


            flow_i21 = flownet(video_ori2, video_ori1)         #计算真实图片2到1的光流
            warp_o1 = flow_warping(video_rep1, flow_i21)       #用光流对生成图片1进行warp，理论上得到生成图片2
            warp_i1 = flow_warping(video_ori1, flow_i21)



            st_loss = F.l1_loss(warp_o1, warp_i1) * args.st #用生成图片2和理论生成图片2相减得到loss
            
            video_e_ = video_e # for visualization
            
            video_e = [pre_process_batch(frame_rep_e,frame_basemask) for frame_rep_e in video_e[1:]]


            video_e1 = torch.stack([video_e[i] for i in range(len(video_e)-2)]).reshape(-1,3,640,640)
            video_e2 = torch.stack([video_e[i] for i in range(1,len(video_e)-1)]).reshape(-1,3,640,640)
            video_e3 = torch.stack([video_e[i] for i in range(2,len(video_e))]).reshape(-1,3,640,640)

            #print (video_e1.shape,video_e2.shape,video_e3.shape)

            flow_21 = flownet(video_e2, video_e1) 
            flow_23 = flownet(video_e2, video_e3) 



            warp_12 = flowBackWarp(video_e1, flow_21)
            warp_32 = flowBackWarp(video_e3, flow_23)



            mask = torch.sigmoid(masknet(torch.cat([warp_12, warp_32], dim=1)))#.repeat(1, 3, 1, 1)

            frame_warpe2s = mask * warp_12 + (1-mask) * warp_32

            
            edit_loss = F.l1_loss(video_e2,frame_warpe2s) * args.edit# * (args.num_continuous-2)

            

                
            
            e_loss = lpips_loss + mse_loss + st_loss + edit_loss# + reg_loss




            optimizer.zero_grad()
            e_loss.backward()
            optimizer.step()


            time2 = time.time()
            
            tdata = time1-time0
            tenc = time2-time1



            pbar.set_description(
                (
                f'Total_iter:{total_iter:010d}; Tdata:{tdata:.4f}; Tenc:{tenc:.4f}; loss:{e_loss.item():.4f}; lpips_loss:{lpips_loss.item():.4f}; mse_loss:{mse_loss.item():.4f}; st_loss:{st_loss.item():.4f}; edit_loss:{edit_loss.item():.4f}'
                )
            )
            
            if args.tensorboard:
                logger.add_scalar('total', e_loss.item(), total_iter)
                logger.add_scalar('lpips', lpips_loss.item(), total_iter)
                logger.add_scalar('mse', mse_loss.item(), total_iter)
                logger.add_scalar('st', st_loss.item(), total_iter)
                logger.add_scalar('edit', edit_loss.item(), total_iter)




            iter_seq = 100

            if total_iter%iter_seq==0:
                with torch.no_grad():
                    video_p = (torch.stack(video_p)).reshape(-1,3,frame_ori1.shape[2], frame_ori1.shape[3])
                    video_ori = (torch.stack(video_ori)).reshape(-1,3,frame_ori1.shape[2], frame_ori1.shape[3])
                    video_e_ = (torch.stack(video_e_)).reshape(-1,3,frame_ori1.shape[2], frame_ori1.shape[3])

                    index_ = []
                    for i in range(video_p.shape[0]):
                        if i%args.batch==0:
                            index_.append(i)


                    sample = torch.cat([video_ori.detach()[index_], video_p.detach()[index_]])
                    sample = torch.cat([sample, video_e_.detach()[index_].cpu()])                                                                

                    #sample = torch.cat([sample, sample_])
                    
                    utils.save_image(
                        sample,
                        _dirs[0]+f"/{str(total_iter).zfill(6)}_"+attr+'.png',
                        nrow=len(index_),
                        normalize=True,
                        range=(-1, 1),
                    )



                    utils.save_image(
                        frame_rep2_e,
                        _dirs[0]+f"/{str(total_iter).zfill(6)}_"+attr+'_edit_rep.png',
                        normalize=True,
                        range=(-1, 1),
                    )


                    utils.save_image(
                        frame_rep2,
                        _dirs[0]+f"/{str(total_iter).zfill(6)}_rec_rep.png",
                        normalize=True,
                        range=(-1, 1),
                    )


                    index_ = []
                    for i in range(frame_warpe2s.shape[0]):
                        if i%args.batch==0:
                            index_.append(i)
                    # print (index_)       
                    sample1 = frame_warpe2s[index_] 
                    sample2 = video_e2[index_]
                    
                    sample = torch.cat([sample1, sample2])
                    utils.save_image(
                        sample,
                        _dirs[0]+f"/{str(total_iter).zfill(6)}_e_warp.png",
                        normalize=True,
                        nrow=len(index_),
                        range=(-1, 1),
                    )
                    
                    utils.save_image(
                        mask[index_],
                        _dirs[0]+f"/{str(total_iter).zfill(6)}_e_mask.png",
                        normalize=False,
                        nrow=len(index_),
                    )




            if total_iter % 1000 == 0:
                    torch.save(
                        {
                            "e": encoder.state_dict(),
                            'g': generator.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            'n': noises_base,
                            "args": args,
                        },
                        _dirs[1]+f"/model_{str(total_iter).zfill(6)}.pt",
                        )
            if total_iter % 100 == 0:
                    torch.save(
                        {
                            "e": encoder.state_dict(),
                            'g': generator.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            'n': noises_base,
                            "args": args,
                            'iter': total_iter,


                        },
                        _dirs[1]+f"/model.pt",
                        )





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--e_ckpt", type=str, default="/data/xyy/RIGID-A100/exp/ood_spiderverse_gwen/checkpoint/model.pt")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--iter", type=int, default=10000001)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--noise_res", type=int, default=32)
    parser.add_argument("--local_rank", type=int, default=0)


    parser.add_argument("--lpips", type=float, default=0.8)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--flow", type=float, default=2.0)
    parser.add_argument("--offset", type=float, default=2.0)

    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--id", type=float, default=0.1)
    parser.add_argument("--adv", type=float, default=0.008)  

    parser.add_argument("--st", type=float, default=2.0)
    parser.add_argument("--lt", type=float, default=2.0)  

    parser.add_argument("--edit", type=float, default=5.0)

    parser.add_argument("--reg", type=float, default=0.5)



    parser.add_argument("--tensorboard", action="store_true",default=True)



    
    args = parser.parse_args()

    device = args.device

    args.start_iter = 0
    # args.list = "./text/CELEBV-HQ-800-Train.txt"
    # args.list = "./text/CELEBV-HQ-800-Test.txt"
    args.list = './text/ood_spiderverse_gwen.txt'
    # args.list = './text/Pixar.txt'

    
    args.num_continuous = 5
    args.epoch = 1000


    args.size = 256
    args.latent = 512
    args.n_mlp = 8
    args.channel_multiplier = 2


    args.rgb_max = 1.0
    args.fp16 = False

    args.frame_skip = 1


    channels = {
    4: 512,
    8: 512 ,
    16: 512 ,
    32: 512 ,
    64: 512 ,
    128: 256 ,
    256: 128,
    }

    noises_base = []
    for i in range(7):
        res = 2**(i+2)
        noises_base.append(torch.randn(1,channels[res],res,res).to(device))
    noises_base.reverse()
    noises_base = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noises_base))[::-1][1:]

    encoder = GPEN_LSTM_Encoder(input_channel=12,size=args.size).to(device)
    
    # generator_init = Generator(args.size,args.latent,args.n_mlp).to(device)
    # weight_ckpt = torch.load('./weight_files/stylegan2-ffhq-256x256.pt', map_location=torch.device('cpu'))["g_ema"]
    # generator_init.load_state_dict(weight_ckpt)

    generator = Generator(args.size,args.latent,args.n_mlp).to(device)
    weight_ckpt = torch.load('./weight_files/stylegan2-ffhq-256x256.pt', map_location=torch.device('cpu'))["g_ema"]
    generator.load_state_dict(weight_ckpt)

    base_dir  = './exp/ood_spiderverse_gwen/'

    # base_dir  = './exp/Pixar/'
    

    _dirs = [base_dir+'/sample/', base_dir+'/checkpoint/', base_dir+'/log/']

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


    dataset = Video_Continuous(args.list,args.num_continuous,frame_skip=args.frame_skip, transform=to_tensor,transform_256=to_tensor_256,transform_1024=to_tensor_1024)
    
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=False,
            num_workers=1,
            pin_memory=False
        )



    flow_warping = Resample2d().to(device)
    flownet = networks.FlowNet2(args)
    checkpoint = torch.load("./weight_files/FlowNet2_checkpoint.pth.tar")
    flownet.load_state_dict(checkpoint['state_dict'])
    flownet = flownet.to(device)


    masknet = UNet(6,1).to(device)
    checkpoint = torch.load('./exp/CELEBV-HQ-Selected/visiblenet-slomo/checkpoint/model_100000.pt')
    masknet.load_state_dict(checkpoint['masknet'])
    masknet = masknet.to(device)

    flowBackWarp = backWarp(640,640,device)

    segmodel = BiSeNet(19).eval().to(device).requires_grad_(False)
    segmodel.load_state_dict(torch.load("./weight_files/79999_iter.pth",map_location=torch.device('cpu')))

    edit_model = F_mapping(mapping_lrmul= 1, mapping_layers=14, mapping_fmaps=512, mapping_nonlinearity = 'linear').to(device)
    


    optimizer = optim.Adam([{'params': encoder.parameters()},
                            {'params': generator.parameters(),}], lr=args.lr)

    encoder = torch.nn.DataParallel(encoder)
    generator = torch.nn.DataParallel(generator)
    # generator_init = torch.nn.DataParallel(generator_init)
    flownet = torch.nn.DataParallel(flownet)
    segmodel = torch.nn.DataParallel(segmodel)
    edit_model = torch.nn.DataParallel(edit_model)
    masknet = torch.nn.DataParallel(masknet)

     
    if args.e_ckpt is not None:
        print("resume training:", args.e_ckpt)
        e_ckpt = torch.load(args.e_ckpt,  map_location=torch.device('cpu'))
        encoder.load_state_dict(e_ckpt["e"])
        generator.load_state_dict(e_ckpt["g"])
        noises_base = e_ckpt["n"]
        # args.start_iter = e_ckpt['iter']

    # for i,j in enumerate(noises_base):
    #     noises_base[i] =  noises_base[i].to(device)


    train(args, train_loader, encoder, generator, flownet, segmodel, edit_model, masknet, flow_warping, flowBackWarp, optimizer, noises_base, device)
