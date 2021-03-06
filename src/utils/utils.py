import torch
import cv2
import numpy as np
from azureml.core import Run
from data.matterport import MatterportDataset


def make_depth_image(img, baseline, color_map=cv2.COLORMAP_JET):
    if torch.is_tensor(img):
        img = img.numpy()
    
    img = np.squeeze(img)

    img = img * (255 / baseline)  # normalize to range {0, 255}
    img_color = cv2.applyColorMap(img.astype(np.uint8), color_map)
    no_info = np.where(img < 1.0)  # indices of pixels where there is no depth info
    img_color[no_info] = [0, 0, 0]  # set color to pitch black

    return img_color


def make_rgb_image(img):
    img = np.moveaxis(img, 0, -1) * 255.0
    return img


def log_output_example(run, x_sparse, x_color, y, output_example, name):
    batch_size = x_sparse.shape[0]
    x_sparse = x_sparse.cpu().detach().numpy()
    x_color = MatterportDataset.inverse_color_transform(x_color)
    x_color = x_color.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    output_example = output_example.cpu().detach().numpy()
    for i in range(batch_size):
        norm_baseline = np.max(np.squeeze(y[i]))
        x_sparse_img = make_depth_image(x_sparse[i], norm_baseline)
        x_color_img = make_rgb_image(x_color[i])
        y_img = make_depth_image(y[i], norm_baseline)
        output_example_img = make_depth_image(output_example[i], norm_baseline)

        output_path = name + '_example' + str(i) + '.png'

        img_depth_sparse = np.hstack([y_img, output_example_img])
        img_out_color = np.hstack([x_sparse_img, x_color_img])
        img_all = np.vstack([img_depth_sparse, img_out_color])
        cv2.imwrite(output_path, img_all)

        run.log_image(name=name + '_Example' + str(i),
                      path=output_path,
                      plot=None,
                      description=name + '_Example' + str(i))
