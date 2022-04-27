from tkinter import E
from torch import get_file_path
from torch.utils.data import Dataset
from typing import List
import numpy as np
import cv2 as cv
from pathlib import Path
import glob
import os
import torchvision
from os.path import exists

class MatterportDataset(Dataset):
    def __init__(self, data_path, N = None):
        self.filepaths = self.get_file_paths(data_path)
        #if N is not None:
        #    self.filepaths = self.filepaths[0:N]
        self.color_normalization = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                    ),])        

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        dense = cv.imread(self.filepaths[idx][0], cv.IMREAD_ANYDEPTH).astype(float)
        dense = cv.resize(dense, (0,0), fx=0.25, fy=0.25)
        dense = dense/4000
        color = cv.imread(self.filepaths[idx][1], cv.IMREAD_COLOR).astype(np.uint8)
        color = cv.resize(color, (0,0), fx=0.25, fy=0.25)
        #print(np.min(color), np.mean(color), np.max(color))
        #color = np.moveaxis(color, -1, 0)
        color = self.color_normalization(color) 
        
        mask_missing_data = dense != 0
        ones = np.ones(dense.shape, dtype=int)
        mask_sparse = np.random.binomial(ones, 0.01)
        sparse = mask_sparse*dense
        mask = mask_missing_data*(np.invert(mask_sparse>0))
        
        sparse = np.expand_dims(sparse, axis=0)
        dense = np.expand_dims(dense, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return sparse, color, dense, mask
        
        
        
    def get_file_paths(self, data_path : str) -> List[str]:
        filepaths = []
        for p in Path(data_path).rglob('*.png'):
            depth = str(p)
            color = depth.replace('.png', '.jpg').replace('undistorted_depth_images', 'undistorted_color_images').replace('_d', '_i')
            if(exists(color) and exists(depth)):
                filepaths.append((depth, color))
            
        return filepaths
    
    @staticmethod
    def inverse_color_transform(images):
        inv_color_normalization = torchvision.transforms.Compose([
                                    torchvision.transforms.Normalize(
                                        mean=[0, 0, 0],
                                        std=[1/0.229, 1/0.224, 1/0.225],
                                    ),
                                    torchvision.transforms.Normalize(
                                        mean=[-0.485, -0.456, -0.406],
                                        std=[1., 1., 1.]
                                    )])
        
        
        return inv_color_normalization(images)*255