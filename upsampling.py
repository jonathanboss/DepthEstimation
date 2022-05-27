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


img_depth = cv2.imread(
    "datasets/matterport_undistorted2/subset/undistorted_depth_images/0bdadf346eb441c48a58d9b8797ee882_d1_3.png", cv2.IMREAD_ANYDEPTH).astype(float)
img_dense_color = make_depth_image(img_depth.copy())

# Make a sparse image
ones = np.ones(img_depth.shape, dtype=int)
mask_sparse = np.random.binomial(ones, 0.01)

img_sparse = mask_sparse * img_depth.copy()
img_sparse_color = make_depth_image(img_sparse)

mask = np.array(mask_sparse, dtype=np.uint8)
mask[np.where(mask_sparse == 1)] = 0
mask[np.where(mask_sparse == 0)] = 1
mask[np.where(img_depth < 1.0)] = 1
# img_upsample = cv2.inpaint(img_sparse_color, mask, 3, cv2.INPAINT_NS)

grid_x, grid_y = np.mgrid[0:mask_sparse.shape[0], 0:mask_sparse.shape[1]]
points = np.array(np.where(img_sparse > 1.0))
values = img_sparse[np.where(img_sparse > 1.0)]
grid_z0 = griddata(points.T, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points.T, values, (grid_x, grid_y), method='linear')
nearest_color = make_depth_image(grid_z0)
linear_color = make_depth_image(np.nan_to_num(grid_z1, nan=0.0))

plt.subplot(221)
plt.imshow(cv2.cvtColor(img_dense_color, cv2.COLOR_BGR2RGB))
plt.title('GT')
plt.axis("off")
plt.subplot(222)
plt.imshow(cv2.cvtColor(img_sparse_color, cv2.COLOR_BGR2RGB))
plt.title('Sparse')
plt.axis("off")
plt.subplot(223)
plt.imshow(cv2.cvtColor(nearest_color, cv2.COLOR_BGR2RGB))
plt.title('Nearest')
plt.axis("off")
plt.subplot(224)
plt.imshow(cv2.cvtColor(linear_color, cv2.COLOR_BGR2RGB))
plt.title('Linear')
plt.gcf().set_size_inches(6, 6)
plt.axis("off")
plt.show()

# cv2.imwrite("dense.png", img_dense_color)
# cv2.imwrite("sparse.png", img_sparse_color)
# cv2.imwrite("upsampled.png", img_upsample)

# cv2.imshow("dense", img_dense_color)
# cv2.waitKey()
#
# cv2.imshow("sparse", img_sparse_color)
# cv2.waitKey()


# cv2.imshow("upsampled", img_upsample)
# cv2.waitKey()