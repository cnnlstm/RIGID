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

from torch import linalg as LA
import itertools
from tensorboardX import SummaryWriter
from utils import *
from util import *


from models.mask_model import *



from criteria.lpips.lpips import LPIPS


# from pwcnet_offical.models.PWCNet import PWCDCNet

import networks

from criteria.lpips.lpips import LPIPS


try:
    from networks.resample2d_package.resample2d import Resample2d
except:
    from .networks.resample2d_package.resample2d import Resample2d



vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22]).cuda()
for param in vgg16_conv_4_3.parameters():
        param.requires_grad = False

# loss function
def lossfn(output, IT):
    It_warp = output

    recnLoss = 204 * F.l1_loss(It_warp, IT)
    prcpLoss = 0.005 * F.mse_loss(vgg16_conv_4_3(It_warp), vgg16_conv_4_3(IT))

    loss = recnLoss + prcpLoss

    return loss, recnLoss,  prcpLoss,




class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut




def train(args, train_loader, masknet, flownet, FlowBackWarp, optimizer, device):
    logger = SummaryWriter(logdir=_dirs[2])

    flownet.eval()

    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    centercrop = transforms.CenterCrop(640)

    train_loader = sample_data(train_loader)
    
    for idx in pbar:
        total_iter = idx + args.start_iter

        if total_iter > args.iter:
            print("Done!")
            break

        batch = next(train_loader)
        frames,_ = batch
        frames = frames.to(args.device)


        #print (frames.shape)
        frame_0 = frames[:,0]
        frame_1 = frames[:,1]
        frame_2 = frames[:,2]
        #print (frame_0.shape)

        flow_10 = flownet(frame_1, frame_0) 
        flow_12 = flownet(frame_1, frame_2) 


        warp_01 = FlowBackWarp(frame_0, flow_10)
        warp_21 = FlowBackWarp(frame_2, flow_12)


        mask = torch.sigmoid(masknet(torch.cat([warp_01, warp_21], dim=1)))#.repeat(1, 3, 1, 1)

        frame_warp1 = mask * warp_01 + (1-mask) * warp_21

        total_loss, recnLoss,  prcpLoss = lossfn(frame_warp1,frame_1)


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        pbar.set_description(
            (
            f'Total_iter:{total_iter:04d}; loss:{total_loss.item():.4f}; prcpLoss:{prcpLoss.item():.4f}; recnLoss:{recnLoss.item():.4f}'
            )
        )


        if args.tensorboard:
            logger.add_scalar('total', total_loss.item(), total_iter)
            logger.add_scalar('prcpLoss', prcpLoss.item(), total_iter)
            logger.add_scalar('recnLoss', recnLoss.item(), total_iter)





        if (total_iter-args.start_iter) <1000:
                
            iter_seq = 50
        else:
            iter_seq = 1000

        if total_iter%iter_seq==0:
            with torch.no_grad():
                sample = torch.cat([frame_1.detach()[:5], frame_warp1.detach()[:5]])
                sample = torch.cat([sample,warp_01[:5]])
                sample = torch.cat([sample,warp_21[:5]])


                utils.save_image(
                    sample,
                    _dirs[0]+f"/{str(total_iter).zfill(6)}.png",
                    nrow=5,
                    normalize=True,
                    range=(-1, 1),
                )


                utils.save_image(
                    mask[:5],
                    _dirs[0]+f"/{str(total_iter).zfill(6)}_mask.png",
                    normalize=False,
                    # range=(-1, 1),
                )



        if total_iter % 10000 == 0:
                torch.save(
                    {
                        "masknet": masknet.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": args,
                    },
                    _dirs[1]+f"/model_{str(total_iter).zfill(6)}.pt",
                    )





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--e_ckpt", type=str, default='/nvme/xyy/VIDEO/RIGID/exp/CELEBV-HQ-Selected/visiblenet-slomo/checkpoint/model_130000.pt')

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--iter", type=int, default=200001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--local_rank", type=int, default=0)


    parser.add_argument("--lpips", type=float, default=0.8)
    parser.add_argument("--edit", type=float, default=5.0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--flow", type=float, default=2.0)
    parser.add_argument("--offset", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--id", type=float, default=0.1)
    parser.add_argument("--adv", type=float, default=0.5)  

    parser.add_argument("--st", type=float, default=2.0)
    parser.add_argument("--lt", type=float, default=2.0)  





    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--tensorboard", action="store_true",default=True)
    
    args = parser.parse_args()
    args.start_iter = 0

    args.size = 256
    args.latent = 512
    args.n_mlp = 5
    args.channel_multiplier = 2
    args.rgb_max = 1.0
    args.fp16 = False


    device = args.device



    flow_warping = Resample2d().to(device)
    flownet = networks.FlowNet2(args)
    checkpoint = torch.load("./weight_files/FlowNet2_checkpoint.pth.tar")
    flownet.load_state_dict(checkpoint['state_dict'])
    flownet = flownet.to(device)


    masknet = UNet(6,1).to(device)

    FlowBackWarp = backWarp(640,640,device)
    args.list =  "./text/CELEBV-HQ-Selected-Full.txt"
    args.num_continuous = 3


    _dirs = ['./exp/CELEBV-HQ-Selected/visiblenet-slomo/sample/','./exp/CELEBV-HQ-Selected/visiblenet-slomo/checkpoint/', './exp/CELEBV-HQ-Selected/visiblenet-slomo/log/']
    
    for x in _dirs:
        try:
            os.makedirs(x)
        except:
            pass
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset = Frame_Continuous(args.list,args.num_continuous,transform=to_tensor)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=False,
            num_workers=1
        )


    optimizer = optim.Adam(masknet.parameters(),
                lr=args.lr,
                betas=(0.9, 0.99))

     
    if args.e_ckpt is not None:
        print("resume training:", args.e_ckpt)
        e_ckpt = torch.load(args.e_ckpt,  map_location=torch.device('cpu'))
        masknet.load_state_dict(e_ckpt["masknet"])
        optimizer.load_state_dict(e_ckpt["optimizer"])

        try:
            ckpt_name = os.path.basename(args.e_ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name.split('_')[1])[0])
        except ValueError:
            pass   


    train(args, train_loader, masknet, flownet, FlowBackWarp, optimizer, device)

