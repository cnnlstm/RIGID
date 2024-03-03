import torch.utils.data as data
import torch
from PIL import Image, ImageFilter
import os
import os.path
import numpy as np
from numpy.random import randint
from torchvision import transforms, utils
import cv2
import torchvision.transforms.functional as tF
import random
import time
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])-1


class Video_Continuous(data.Dataset):

    def __init__(self, list_file,
                 num_continuous, 
                 modal='train',image_tmpl='frame{:04d}.jpg',mask_tmpl='mask{:04d}.jpg',code_tmpl='frame{:04d}.npy', frame_skip=1, transform=None, transform_256=None,transform_1024=None,
                 ):

        self.list_file = list_file
        self.num_continuous = num_continuous
        self.image_tmpl = image_tmpl
        self.mask_tmpl = mask_tmpl
        self.code_tmpl = code_tmpl
        self.transform = transform
        self.frame_skip = frame_skip

        self.transform_256 = transform_256
        self.transform_1024 = transform_1024
        self.modal = modal


        self._parse_list()

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx)))

    def _load_mask(self, directory, idx):
        return cv2.imread(os.path.join(directory, self.mask_tmpl.format(idx)))

    def _load_code(self, directory, idx):
        return np.load(os.path.join(directory, self.code_tmpl.format(idx))).squeeze()


    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]


    def _get_val_indices(self, record):
        if self.modal == 'train':
            start_index = randint(0,record.num_frames - self.num_continuous * self.frame_skip)
            offsets = [int(start_index + (x * self.frame_skip)) for x in range(self.num_continuous)]
        else:
            offsets = [x for x in range(record.num_frames)]

        return offsets

    def _cv2_to_pil(self,open_cv_image):
        return Image.fromarray(open_cv_image[:, :, ::-1].copy())

    def _find_coeffs(self, pa, pb):

        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def __getitem__(self, index):
        record = self.video_list[index]
        continuous_indices = self._get_val_indices(record)
        return self.get(record, continuous_indices)





    def get(self, record, indices):
        video_ori = []
        video_256 = []
        #video_1024 = []
        video_rec = []
        #video_edit = []


        video_code = []
        video_coeff = []
        video_crop_size = []
        video_center = []
        video_quad_0 = []

        # time0 = time.time()
        stat_dict = np.load(record.path + 'stat_dict.npy', allow_pickle=True).item()

        for ix, ind in enumerate(indices):   
            # print ('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', record.path)         
            img = self._load_image(record.path, ind)
            img_ori = self._load_image(record.path.replace("frame_aligned","frame"), ind)
            img_ori = self.transform(img_ori)

            img_256 = self.transform_256(img)
            #img_1024 = self.transform(img)


            img_rec = self._load_image(record.path.replace("frame_aligned","frame_aligned_rec_256"), ind)
            img_rec = self.transform(img_rec)

            # img_edit = self._load_image(record.path.replace("frame_aligned","frame_aligned_edit_age"), ind)
            # img_edit = self.transform_1024(img_edit)


            code = self._load_code(record.path.replace("frame_aligned","frame_aligned_latent_256"), ind)

            quad_f = stat_dict['quad'][ind]
            quad_0 = stat_dict['crop'][ind]

            coeff = self._find_coeffs([(quad_f[0], quad_f[1]), (quad_f[2] , quad_f[3]), (quad_f[4], quad_f[5]), (quad_f[6], quad_f[7])],
            [(0, 0), (0, 1024), (1024, 1024), (1024, 0)])
            
            crop_size = (quad_0[3] - quad_0[1], quad_0[2] - quad_0[0])
            #crop_size_pil = (quad_0[2] - quad_0[0], quad_0[3] - quad_0[1])

            '''
            mask = self._load_mask(record.path.replace("frame_aligned","frame"), ind) # inner-face region
            mask = cv2.dilate(mask, np.ones((10,10), np.uint8), iterations=5)
            mask = self._cv2_to_pil(mask).filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
            mask = np.array(mask)[:, :, np.newaxis]/255.
            '''

            #mask = self._load_mask(record.path.replace("frame_aligned","frame"), ind)[:,:,0]/255. # inner-face region
            if ix == 0:
                mask = self._load_mask(record.path.replace("frame_aligned","frame"), ind)[:,:,0]
                max_ = np.max(mask)
                indexs = np.where(mask==max_)
                mask_x = int(indexs[0].mean())
                mask_y = int(indexs[1].mean())
                mask_center = [mask_x,mask_y]
                video_center.append(mask_center)
            '''
            # 0: 'background' 1: 'skin'   2: 'nose'
            # 3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
            # 6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
            # 9: 'r_ear'  10: 'mouth' 11: 'u_lip'
            # 12: 'l_lip' 13: 'hair'  14: 'hat'
            # 15: 'ear_r' 16: 'neck_l'  17: 'neck'
            # 18: 'cloth'

            # face_part_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
            # face_part_ids = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]

            parsing = self._load_mask(record.path.replace("frame_aligned","frame_aligned_parsing"), ind)            
            mask = Image.fromarray(np.zeros((720,1280))*255)
            mask_ = np.ones_like(parsing)*255
            dark_parts = [0, 16, 17, 18]
            for face_id in dark_parts:
                mask_index = np.where(parsing==face_id)
                mask_[mask_index] = 0
            mask_ = cv2.dilate(mask_, np.ones((10,10), np.uint8), iterations=5)
            mask_ = self._cv2_to_pil(mask_).filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
            mask_ = mask_.transform(crop_size_pil, Image.PERSPECTIVE, coeff, Image.BICUBIC)
            mask.paste(mask_, (int(quad_0[0]), int(quad_0[1])))
            mask = np.array(mask)[:, :, np.newaxis]/255.
            '''
            
            # print (img_ori)
            video_ori.append(img_ori)
            video_256.append(img_256)
            #video_1024.append(img_1024)
            video_rec.append(img_rec)
            #video_edit.append(img_edit)

            video_code.append(torch.from_numpy(code))
            video_coeff.append(torch.from_numpy(coeff))
            

            video_crop_size.append(crop_size)
            video_quad_0.append(quad_0)
        #time1 = time.time()
        #print ('xxx',time1-time0)
        # return video_ori, video_256, video_1024, video_rec, video_edit, codes, coeffs, masks, crop_sizes, quad_0s, record.path.split("/")[-2],indices
        # print (video_ori[0].shape,video_256[0].shape,video_rec[0].shape,video_coeff[0],video_crop_size[0],video_quad_0[0])
        # print (record.path.split("/")[-2],indices)
        return video_ori,video_256,video_rec,video_code,video_coeff,video_center,video_crop_size,video_quad_0, record.path.split("/")[-2],indices


        # video_ori,video_256, video_1024,video_rec,video_code,video_coeff,video_center,video_crop_size,video_quad_0,video_name,video_slices
        
        # frame_ori1 = video_ori[t-1].to(device)
        # frame_i1 = video_256[t-1].to(device)
        # frame_rec1 = video_rec[t-1].to(device)
        # coeff_1 = video_coeff[t-1].to(device)
        # crop_size_1 = video_crop_size[t-1]
        # quad_1 = video_quad_0[t-1]
        # code_i1 = video_code[t-1].to(device)


    def __len__(self):
        return len(self.video_list)






class Frame_Continuous(data.Dataset):

    def __init__(self, list_file,
                 num_continuous, image_tmpl='frame{:04d}.jpg', transform=None,
                 ):
        self.list_file = list_file
        self.num_continuous = num_continuous
        self.image_tmpl = image_tmpl
        self.transform = transform
        self._parse_list()

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx)))

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _get_val_indices(self, record):
        start_index = randint(0,record.num_frames - self.num_continuous)
        offsets = [int(start_index + x) for x in range(self.num_continuous)]
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        continuous_indices = self._get_val_indices(record)
        return self.get(record, continuous_indices)

    def get(self, record, indices):
        video_ori = []

        for ind in indices:            
            img = self._load_image(record.path, ind)
            img_ori = self._load_image(record.path.replace("frame_aligned","frame"), ind)
            img_ori = self.transform(img_ori)
            video_ori.append(img_ori)
            
        return torch.stack(video_ori), indices

    def __len__(self):
        return len(self.video_list)


