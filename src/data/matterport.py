import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np
import cv2 as cv
from pathlib import Path
from torchvision import transforms
from os.path import exists


class MatterportDataset(Dataset):
    def __init__(self, data_path, normalize=False):
        self.filepaths = self.get_file_paths(data_path)
        self.normalize = normalize

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load dense depth image
        dense = cv.imread(self.filepaths[idx][0], cv.IMREAD_ANYDEPTH).astype(float)
        dense = dense / 2**16  # normalize

        color = cv.imread(self.filepaths[idx][1], cv.IMREAD_COLOR)
        assert (np.max(color) <= 255), f'Max value in image {np.max(color)}. Should be lower than 255'

        new_size = (dense.shape[0] // 4, dense.shape[1] // 4)

        transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        ])

        # transform RGB image
        color = transform_rgb(color)
        if self.normalize:
            color = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(color)

        # transform depth image
        dense = transform_rgb(dense)

        # Make a sparse image
        ones = np.ones(dense.shape, dtype=int)
        mask_sparse = torch.from_numpy(np.random.binomial(ones, 0.01))
        sparse = mask_sparse * dense
        
        # make binary mask of where sparse depth info exists
        validity_idx = torch.nonzero(sparse, as_tuple=True)
        validity_mask = torch.zeros(sparse.shape)
        validity_mask[validity_idx] = 1

        return color, sparse, validity_mask, dense

    @staticmethod
    def get_file_paths(data_path: str) -> List[str]:
        filepaths = []
        for p in Path(data_path).rglob('*.png'):
            depth = str(p)
            color = depth.replace('.png', '.jpg').replace('undistorted_depth_images',
                                                          'undistorted_color_images').replace('_d', '_i')
            if exists(color) and exists(depth):
                filepaths.append((depth, color))
        return filepaths[1000:11000]
