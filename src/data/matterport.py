import torchvision
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np
import cv2 as cv
from pathlib import Path
from torchvision import transforms
from os.path import exists


class MatterportDataset(Dataset):
    def __init__(self, data_path, sparsity_level, normalize=True):
        self.filepaths = self.get_file_paths(data_path)
        self.normalize = normalize
        self.sparsity_level = sparsity_level

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load dense depth image
        dense = cv.imread(self.filepaths[idx][0], cv.IMREAD_ANYDEPTH).astype(float)
        dense = dense / 2 ** 16  # normalize

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
        mask_sparse = torch.from_numpy(np.random.binomial(ones, self.sparsity_level))
        sparse = mask_sparse * dense

        # make binary mask of where sparse depth info exists
        validity_idx = torch.nonzero(sparse, as_tuple=True)
        validity_mask = torch.zeros(sparse.shape)
        validity_mask[validity_idx] = 1

        return color, sparse, validity_mask, dense

    @staticmethod
    def inverse_color_transform(images):
        inv_color_normalization = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[0, 0, 0],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
            torchvision.transforms.Normalize(
                mean=[-0.485, -0.456, -0.406],
                std=[1., 1., 1.]
            )])

        return inv_color_normalization(images)

    @staticmethod
    def get_file_paths(data_path: str):
        filepaths = []
        for p in Path(data_path).rglob('*.png'):
            depth = str(p)
            color = depth.replace('.png', '.jpg').replace('undistorted_depth_images',
                                                          'undistorted_color_images').replace('_d', '_i')
            if exists(color) and exists(depth):
                filepaths.append((depth, color))
        return filepaths


# import torchvision
# import torch
# from torch.utils.data import Dataset
# from typing import List, Tuple
# import numpy as np
# import cv2 as cv
# from pathlib import Path
# from torchvision import transforms
# from os.path import exists
# import cv2 as cv2
#
#
# class MatterportDataset(Dataset):
#     def __init__(self, data_path, normalize=True):
#         self.filepaths = self.get_file_paths(data_path)
#         self.normalize = normalize
#         self.detector = cv2.ORB_create(edgeThreshold=10)
#
#     def __len__(self):
#         return len(self.filepaths)
#
#     def __getitem__(self, idx):
#         # Load dense depth image
#         dense = cv.imread(self.filepaths[idx][0], cv.IMREAD_ANYDEPTH).astype(float)
#         dense = dense / 2 ** 16  # normalize
#
#         color = cv.imread(self.filepaths[idx][1], cv.IMREAD_COLOR)
#         assert (np.max(color) <= 255), f'Max value in image {np.max(color)}. Should be lower than 255'
#
#         new_size = (dense.shape[0] // 4, dense.shape[1] // 4)
#
#         transform_rgb = transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Resize(size=new_size, interpolation=transforms.InterpolationMode.BILINEAR)
#         ])
#         color = cv2.resize(color, (0, 0), 0, 0.25, 0.25)
#
#         # cv2.imshow('color', cv2.resize(color.copy(), (0,0), 0, 0.5, 0.5))
#         # transform RGB image
#         # color = transform_rgb(color)
#         # cv2.imshow('color', color)
#
#         kps = self.detector.detect(color, None)
#         # print(len(kps))
#         kp_mask = np.zeros(new_size, np.uint8)
#         for kp in kps:
#             kp_mask[int(kp.pt[1])][int(kp.pt[0])] = 1
#
#         # cv2.imshow('kp', kp_mask*255)
#         # cv2.waitKey(0)
#
#         color = transform_rgb(color)
#         if self.normalize:
#             color = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(color)
#
#         # transform depth image
#         dense = cv2.resize(dense, (0, 0), 0, 0.25, 0.25)
#         dense = transform_rgb(dense)
#
#         # Make a sparse image
#         # ones = np.ones(dense.shape, dtype=int)
#         # mask_sparse = torch.from_numpy(np.random.binomial(ones, 0.01))
#         sparse = torch.from_numpy(kp_mask) * dense
#
#         # make binary mask of where sparse depth info exists
#         validity_idx = torch.nonzero(sparse, as_tuple=True)
#         validity_mask = torch.zeros(sparse.shape)
#         validity_mask[validity_idx] = 1
#
#         return color, sparse, validity_mask, dense
#
#     @staticmethod
#     def inverse_color_transform(images):
#         inv_color_normalization = torchvision.transforms.Compose([
#             torchvision.transforms.Normalize(
#                 mean=[0, 0, 0],
#                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
#             ),
#             torchvision.transforms.Normalize(
#                 mean=[-0.485, -0.456, -0.406],
#                 std=[1., 1., 1.]
#             )])
#
#         return inv_color_normalization(images)
#
#     @staticmethod
#     def get_file_paths(data_path: str) -> List[str]:
#         filepaths = []
#         for p in Path(data_path).rglob('*.png'):
#             depth = str(p)
#             color = depth.replace('.png', '.jpg').replace('undistorted_depth_images',
#                                                           'undistorted_color_images').replace('_d', '_i')
#             if exists(color) and exists(depth):
#                 filepaths.append((depth, color))
#         return filepaths
#
#
# def main():
#     data_path = 'datasets/matterport_undistorted2/'
#     dataset = MatterportDataset(data_path)
#     for i in range(30, 100):
#         color, sparse, validity_mask, dense = dataset.__getitem__(i)
#
#
# if __name__ == '__main__':
#     main()