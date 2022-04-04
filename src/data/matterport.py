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
    def __init__(self, data_path):
        self.filepaths = self.get_file_paths(data_path)[0:4]
        self.color_normalization = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                    ),])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        dense = cv.imread(self.filepaths[idx][0], cv.IMREAD_GRAYSCALE).astype(float)
        color = cv.imread(self.filepaths[idx][1], cv.IMREAD_COLOR).astype(float)
        color_shape = color.shape
        color = self.color_normalization(color)
        color = color.numpy()
        color = color.reshape((3, color_shape[0], color_shape[1]))

        mask_missing_data = dense != 0

        ones = np.ones(dense.shape, dtype=int)
        mask_sparse = np.random.binomial(ones, 0.1)
        sparse = mask_sparse*dense

        mask = mask_missing_data*(np.invert(mask_sparse>0))
        sparse = np.expand_dims(sparse, axis=0)
        color_depth_4channel = np.vstack([color, sparse])
        dense = np.expand_dims(dense, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return color_depth_4channel, dense, mask
        
        
        
    def get_file_paths(self, data_path : str) -> List[str]:
        filepaths = []
        for p in Path(data_path).rglob('*.png'):
            depth = str(p)
            color = depth.replace('.png', '.jpg').replace('matterport_depth_images', 'matterport_color_images').replace('_d', '_i')
            if(exists(color) and exists(depth)):
                filepaths.append((depth, color))
            
        return filepaths