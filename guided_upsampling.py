import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from scipy.interpolate import griddata
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def make_depth_image(img, color_map=cv2.COLORMAP_JET):
    img = np.squeeze(img)
    img = img * (255 / np.max(img))  # normalize to range {0, 255}
    no_info = np.where(img < 1.0)  # indices of pixels where there is no depth info
    img[no_info] = 0  # set color to pitch black
    img_color = cv2.applyColorMap(img.astype(np.uint8), color_map)

    return img_color

# def nearest_nonzero_idx(a,x,y):
#     tmp = a[x,y]
#     a[x,y] = 0
#     r,c = np.nonzero(a)
#     a[x,y] = tmp
#     min_idx = ((r - x)**2 + (c - y)**2).argmin()
#     return r[min_idx], c[min_idx]

def nearest_nonzero_idx(a,x,y):
    idx = np.argwhere(a)
    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x,y]).all(1)]
    return tuple(idx[((idx - [x,y])**2).sum(1).argmin()])

# Load images
img_depth = cv2.imread(
    "datasets/matterport_undistorted2/subset/undistorted_depth_images/00ebbf3782c64d74aaf7dd39cd561175_d0_0.png", cv2.IMREAD_ANYDEPTH).astype(float)
dim = (img_depth.shape[1] // 4, img_depth.shape[0] // 4)
img_depth = cv2.resize(img_depth, dim, interpolation=cv2.INTER_NEAREST)
img_dense_color = make_depth_image(img_depth.copy())

# Make a sparse image
ones = np.ones(img_depth.shape, dtype=int)
mask_sparse = np.random.binomial(ones, 0.01)

img_sparse = mask_sparse * img_depth.copy()
img_sparse_color = make_depth_image(img_sparse)

upsampled = np.zeros_like(img_sparse)

# filtering
kernel = (3, 3)
padding = (kernel[0] // 2, kernel[1] // 2)
padded = np.pad(img_sparse, ((padding[0], padding[0]), (padding[1], padding[1])), 'symmetric')
for x in range(padding[0], img_sparse_color.shape[0] + padding[0]):
    for y in range(padding[1], img_sparse_color.shape[1] + padding[1]):
        patch = padded[x - padding[0]:x+padding[0]+1, y - padding[1]:y + padding[1]+1]
        if upsampled[x - padding[0], y - padding[1]] == 0:
            upsampled[x - padding[0], y - padding[1]] = upsampled[nearest_nonzero_idx(img_sparse, x - padding[0], y - padding[1])]

upsampled_color = make_depth_image(upsampled)

plt.subplot(121)
plt.imshow(cv2.cvtColor(img_dense_color, cv2.COLOR_BGR2RGB))
plt.title('GT')
plt.axis("off")
plt.subplot(122)
plt.imshow(cv2.cvtColor(upsampled_color, cv2.COLOR_BGR2RGB))
plt.title('Sparse')
plt.axis("off")
plt.show()