import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import os

class DepthCompletionDataset(Dataset):
    def __init__(self, dataset_dir, img_extension: str = '.png', transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.img_extension = img_extension
        self.img_ids = [Path(file).stem for file in os.listdir(dataset_dir) if Path(file).suffix == img_extension]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, f'{self.img_ids[idx]}{self.img_extension}')
        gt = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ones = np.ones(gt.shape, dtype=int)
        mask = np.random.binomial(ones, 0.01)
        sparse = mask * gt

        validity_idx = np.nonzero(sparse)
        validity_mask = np.zeros(sparse.shape)
        validity_mask[validity_idx] = 1

        # add 1 extra dimension
        sparse = np.expand_dims(sparse, axis=0)
        gt = np.expand_dims(gt, axis=0)
        validity_mask = np.expand_dims(validity_mask, axis=0)

        if self.transform:
            sparse = self.transform(sparse)
            gt = self.transform(gt)
            validity_mask = self.transform(validity_mask)

        return sparse, gt, validity_mask
