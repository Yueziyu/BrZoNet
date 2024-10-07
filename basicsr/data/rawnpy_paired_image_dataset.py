from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import numpy as np
import torch
import glob
import os


class Dataset_PairedImage_Rawnpy(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImage_Rawnpy, self).__init__()
        self.opt = opt
        
        if self.opt["phase"] == "train":
            self.train_fns = glob.glob(self.opt["dataroot_lq"] + '0*_00*.npy')
            self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in self.train_fns]
        elif self.opt["phase"] == "val":
            self.train_fns = glob.glob(self.opt["dataroot_lq"] + '1*_00*.npy')
            self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in self.train_fns]
            
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
            
    def pack_rawnpy(self, im):
        # pack Bayer image to 4 channels
        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]
      
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                            im[0:H:2, 1:W:2, :],
                            im[1:H:2, 1:W:2, :],
                            im[1:H:2, 0:W:2, :]), axis=2)
        return out
    
    def __getitem__(self, index):
        scale = self.opt['scale']
        
        in_path = self.train_fns[index]
        in_fn = os.path.basename(in_path)
        train_id = self.train_ids[index] 
        
        gt_path = glob.glob(self.opt["dataroot_gt"] + '%05d_00*.npy' % train_id)[0]
        gt_fn = os.path.basename(gt_path)
        
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        
        img_lq = np.load(in_path)
        img_gt = np.load(gt_path)
        
        img_lq = self.pack_rawnpy(img_lq) * ratio
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        
        img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': in_path, 'gt_path': gt_path}
    
    def __len__(self):
        return len(self.train_ids)