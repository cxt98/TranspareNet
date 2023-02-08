import os
import torch

import numpy as np
np.random.seed(0)
from PIL import Image
# import OpenEXR
import Imath
import glob
from skimage.transform import resize
import cv2

#TODO: change this into config params
ROOT = "/media/cxt/6358C6357FEBD1E6/clearpose_dataset"



class ClearPose:
    def __init__(
            self, root=ROOT, split="train", transforms=None, processed=False, small_large='small'
    ):
        # Split can be train or test-val
        self.transforms = transforms
        self.split = split
        self.small_large = small_large
        if split == 'train':
            self.set_scenes = [
                [1, 1], [1, 2], [1, 3], [1, 4],
                [4, 1], [4, 2], [4, 3], [4, 4], [4, 5],
                [5, 1], [5, 2], [5, 3], [5, 4], [5, 5],
                [6, 1], [6, 2], [6, 3], [6, 4], [6, 5],
                [7, 1], [7, 2], [7, 3], [7, 4], [7, 5],
            ]
        else:
            self.set_scenes = [
                [4, 6], [5, 6], [6, 6], [7, 6],
                [8, 1], [8, 2], [3, 1], [3, 3]  
            ]
        self.data_root = root
        # self.split_file = os.path.join(root, "splits", split + ".txt")
        # self.data_list = self._get_data_list(self.split_file)
        self.color_name, self.depth_name, self.render_name,self.mask_name = [], [], [], []
        self.normal_name = []
        self._load_data()

    def _load_data(self):
        for set_scene in self.set_scenes:
            path = f'{self.data_root}/set{set_scene[0]}/scene{set_scene[1]}/'
            # collect transparent rgb, mask, depth paths
            cur_image_paths = sorted(glob.glob(os.path.join(path, '*-color.png')) )
            cur_mask_paths = [p.replace('-color.png', '-label.png') for p in cur_image_paths]
            cur_transparent_depth_paths = [p.replace('-color.png', '-depth.png') for p in cur_image_paths]
            cur_opaque_depth_paths = [p.replace('-color.png', '-depth_true.png') for p in cur_image_paths]
            
            self.color_name += cur_image_paths
            self.mask_name += cur_mask_paths
            self.depth_name += cur_transparent_depth_paths
            self.render_name += cur_opaque_depth_paths


    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        # color           = np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255. # exr_loader(self.color_name[index], ndim=3) / 255.  #np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255.
        # render_depth    = exr_loader(self.render_name[index], ndim=1) # np.array(Image.open(self.render_name[index])) / 4000.
        # depth           = exr_loader(self.depth_name[index], ndim=1) #np.array(Image.open(self.depth_name[index])) / 4000.

        color = cv2.imread(self.color_name[index]).transpose([2, 0, 1]) / 255.
        render_depth = cv2.imread(self.render_name[index], -1) / 1000.
        depth = cv2.imread(self.depth_name[index], -1) / 1000.

        # Load the mask
        # mask = png_loader(self.mask_name[index])
        mask = cv2.imread(self.mask_name[index], -1)
        mask[mask != 0] = 1

        # if self.depth_name[index].endswith('depth-rectified.exr'):
        #     # Remove the portion of the depth image with transparent object
        #     # If image is synthetic
        #     depth[np.where(mask==0)] = 0
        
        # Resize arrays:
        
        # color =resize(color, (3,480,640))
        # assert len(render_depth.shape) == 2 , 'There is channel dimension'
        # render_depth = resize(render_depth,(480,640))
        # depth =resize(depth,(480,640))
        # mask = resize(mask,(480,640))

        # render_depth[np.isnan(render_depth)] = 0.0
        # render_depth[np.isinf(render_depth)] = 0.0

        if self.small_large == 'small': # 'large' is default as (480, 640)
            color = resize(color, (3, 240, 320))
            depth = resize(depth, (240, 320))
            render_depth = resize(render_depth, (240, 320))
            mask = resize(mask, (240, 320))


        return  {
            'color':        torch.tensor(color, dtype=torch.float32),
            'raw_depth':    torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
            'mask':         torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            #'normals':      torch.tensor(normals, dtype=torch.float32).unsqueeze(0),
            'gt_depth':     torch.tensor(render_depth, dtype=torch.float32).unsqueeze(0),
        }