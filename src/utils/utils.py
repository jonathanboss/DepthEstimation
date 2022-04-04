import torch
import cv2 as cv
import numpy as np

def make_depth_image(img, window_handle: str = 'image', color_map=cv.COLORMAP_JET):
    if torch.is_tensor(img):
        img = img.numpy()
    
    print(img.shape)
    img = np.squeeze(img)

    no_info = np.where(img < 1.0)  # indices of pixels where there is no depth info
    img_color = cv.applyColorMap(img.astype(np.uint8), color_map)
    img_color[no_info] = [0, 0, 0]  # set color to pitch black
    eps = 10e-8
    img_color = img_color * (255 / (np.max(img_color)+eps))  # normalize to range {0, 255}
    return img_color