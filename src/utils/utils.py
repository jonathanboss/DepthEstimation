import torch
import cv2
import numpy as np
from azureml.core import Run

def make_depth_image(img, color_map=cv2.COLORMAP_JET):
    if torch.is_tensor(img):
        img = img.numpy()
    img = np.squeeze(img)

    no_info = np.where(img < 1.0)  # indices of pixels where there is no depth info
    img_color = cv2.applyColorMap(img.astype(np.uint8), color_map)
    img_color[no_info] = [0, 0, 0]  # set color to pitch black
    img_color = img_color * (255 / np.max(img_color))  # normalize to range {0, 255}
    return img_color

def log_output_example(run, x, y, output_example):
    batch_size = x.shape[0]
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    output_example = output_example.cpu().detach().numpy()

    for i in range(batch_size):
        x_img = make_depth_image(x[i])
        y_img = make_depth_image(y[i])
        output_example_img = make_depth_image(output_example[i])

        output_path = 'output_example' + str(i) + '.png'
        x_path = 'input_example' + str(i) + '.png'
        y_path = 'gt_example' + str(i) + '.png'

        cv2.imwrite(output_path, output_example_img)
        cv2.imwrite(x_path, x_img)
        cv2.imwrite(y_path, y_img)

        run.log_image(name='Example of output',
                      path=output_path,
                      plot=None,
                      description='Example of output')

        run.log_image(name='Example of input',
                      path=x_path,
                      plot=None,
                      description='Example of input')

        run.log_image(name='Example of ground truth',
                      path=y_path,
                      plot=None,
                      description='Example of ground truth')