import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def make_depth_image(img, color_map=cv2.COLORMAP_JET):
    img = np.squeeze(img)
    img = img * (255 / np.max(img))  # normalize to range {0, 255}
    no_info = np.where(img < 1.0)  # indices of pixels where there is no depth info
    img[no_info] = 0  # set color to pitch black
    img_color = cv2.applyColorMap(img.astype(np.uint8), color_map)

    return img_color


def gauss(img, spatialKern, rangeKern):
    gaussianSpatial = 1 / math.sqrt(2 * math.pi * (
                spatialKern ** 2))  # gaussian function to calcualte the spacial kernel ( the first part 1/sigma * sqrt(2π))
    gaussianRange = 1 / math.sqrt(2 * math.pi * (rangeKern ** 2))  # gaussian function to calcualte the range kernel
    matrix = np.exp(-np.arange(256) * np.arange(256) * gaussianRange)
    xx = -spatialKern + np.arange(2 * spatialKern + 1)
    yy = -spatialKern + np.arange(2 * spatialKern + 1)
    x, y = np.meshgrid(xx, yy)
    spatialGS = gaussianSpatial * np.exp(-(x ** 2 + y ** 2) / (2 * (
                gaussianSpatial ** 2)))  # calculate spatial kernel from the gaussian function. That is the gaussianSpatial variable multiplied with e to the power of (-x^2 + y^2 / 2*sigma^2)
    return matrix, spatialGS


def padImage(img, spatialKern):  # pad array with mirror reflections of itself.
    img = np.pad(img, ((spatialKern, spatialKern), (spatialKern, spatialKern)), 'symmetric')
    return img


# def jointBilateralFilter(img, img1, spatialKern, rangeKern):
#     h, w, ch = img.shape  # get the height,width and channel of the image with no flash
#     orgImg = padImage(img, spatialKern)  # pad image with no flash
#     secondImg = padImage(img1, spatialKern)  # pad image with flash
#     matrix, spatialGS = gauss(img, spatialKern, rangeKern)  # apply gaussian function
#
#     outputImg = np.zeros((h, w, ch), np.uint8)  # create a matrix the size of the image
#     summ = 1
#     for x in range(spatialKern, spatialKern + h):
#         for y in range(spatialKern, spatialKern + w):
#             for i in range(0, ch):  # iterate through the image's height, width and channel
#                 # apply the equation that is mentioned in the pdf file
#                 neighbourhood = secondImg[x - spatialKern: x + spatialKern + 1, y - spatialKern: y + spatialKern + 1,
#                                 i]  # get neighbourhood of pixels
#                 central = secondImg[x, y, i]  # get central pixel
#                 res = matrix[abs(neighbourhood - central)]  # subtract them
#                 summ = summ * res * spatialGS  # multiply them with the spatial kernel
#                 norm = np.sum(res)  # normalization term
#                 outputImg[x - spatialKern, y - spatialKern, i] = np.sum(
#                     res * orgImg[x - spatialKern: x + spatialKern + 1, y - spatialKern: y + spatialKern + 1,
#                           i]) / norm  # apply full equation of JBF(img,img1)
#     return outputImg

def jointBilateralFilter(img, img1, spatialKern, rangeKern):
    h, w = img.shape  # get the height,width and channel of the image with no flash
    orgImg = padImage(img, spatialKern)  # pad image with no flash
    secondImg = padImage(img1, spatialKern)  # pad image with flash
    matrix, spatialGS = gauss(img, spatialKern, rangeKern)  # apply gaussian function

    outputImg = np.zeros((h, w), np.uint8)  # create a matrix the size of the image
    summ = 1
    for x in range(spatialKern, spatialKern + h):
        for y in range(spatialKern, spatialKern + w):
            # apply the equation that is mentioned in the pdf file
            neighbourhood = secondImg[x - spatialKern: x + spatialKern + 1, y - spatialKern: y + spatialKern + 1]  # get neighbourhood of pixels
            central = secondImg[x, y]  # get central pixel
            res = matrix[abs(neighbourhood - central)]  # subtract them
            summ = summ * res * spatialGS  # multiply them with the spatial kernel
            norm = np.sum(res)  # normalization term
            outputImg[x - spatialKern, y - spatialKern] = np.sum(
                res * orgImg[x - spatialKern: x + spatialKern + 1, y - spatialKern: y + spatialKern + 1]) / norm  # apply full equation of JBF(img,img1)
    return outputImg


spatialKern = 4
rangeKern = 2

#dense
img_dense = cv2.imread('datasets/matterport_undistorted2/subset/undistorted_depth_images/0bdadf346eb441c48a58d9b8797ee882_d1_3.png', cv2.IMREAD_ANYDEPTH).astype(int)
dim = (img_dense.shape[1] // 4, img_dense.shape[0] // 4)
img_dense = cv2.resize(img_dense, dim, interpolation=cv2.INTER_NEAREST)
#img_dense = make_depth_image(img_dense.copy())

# sparse
ones = np.ones(img_dense.shape, dtype=int)
mask_sparse = np.random.binomial(ones, 0.5)
img_sparse = mask_sparse * img_dense.copy()
#img_sparse = make_depth_image(img_sparse)

# RGB
img_rgb = cv2.imread('datasets/matterport_undistorted2/subset/undistorted_color_images/0bdadf346eb441c48a58d9b8797ee882_i1_3.jpg')
img_rgb = cv2.resize(img_rgb, dim, interpolation=cv2.INTER_NEAREST)
img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#filteredimg = jointBilateralFilter(img_sparse // 255, img_rgb, spatialKern, rangeKern)

filteredimg = img_sparse.copy()
filteredimg = cv2.ximgproc.jointBilateralFilter(img_sparse // 255, img_rgb, 3, sigmaColor=0.5, sigmaSpace=0)
#filteredimg = img_rgb.copy()
# cv2.imshow('input', img)  # show original no flash image
# cv2.imshow('JointBilateralFilter', filteredimg)  # show image after joint bilateral filter is applied

img_dense_color = make_depth_image(img_dense // 255)
img_sparse_color = make_depth_image(img_sparse // 255)
filteredimg_color = make_depth_image(filteredimg)

plt.subplot(221)
plt.imshow(cv2.cvtColor(img_dense_color, cv2.COLOR_BGR2RGB))
plt.title('GT')
plt.axis("off")
plt.subplot(222)
plt.imshow(img_rgb)
plt.title('RGB')
plt.axis("off")
plt.subplot(223)
plt.imshow(cv2.cvtColor(img_sparse_color, cv2.COLOR_BGR2RGB))
plt.title('Sparse')
plt.axis("off")
plt.subplot(224)
plt.imshow(cv2.cvtColor(filteredimg_color, cv2.COLOR_BGR2RGB))
plt.title('Filtered')
plt.gcf().set_size_inches(6, 6)
plt.axis("off")
plt.show()

cv2.imwrite("bf/gt_bf.png", img_dense_color)
cv2.imwrite("bf/sparse_bf.png", img_sparse_color)
cv2.imwrite("bf/filtered_bf.png", filteredimg_color)