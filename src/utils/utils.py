import torch
import cv2
import numpy as np
from azureml.core import Run

def make_depth_image(img, color_map=cv2.COLORMAP_JET):
    if torch.is_tensor(img):
        img = img.numpy()
    
    img = np.squeeze(img)

    no_info = np.where(img == 0.0)  # indices of pixels where there is no depth info
    img = img * 2**8
    img_color = cv2.applyColorMap(img.astype(np.uint8), color_map)
    img_color[no_info] = [0, 0, 0]  # set color to pitch black

    # normalize to range {0, 255}
    return img_color

def log_output_example(run, x_sparse, x_color, y, output_example, name):
    batch_size = x_sparse.shape[0]
    x_sparse = x_sparse.cpu().detach().numpy()
    x_color = x_color.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    output_example = output_example.cpu().detach().numpy()
    for i in range(4):#range(batch_size):
        x_sparse_img = make_depth_image(x_sparse[i])
        x_color_img = np.moveaxis(x_color[i], 0, -1)
        y_img = make_depth_image(y[i])
        output_example_img = make_depth_image(output_example[i])

        output_path = name + '_example' + str(i) + '.png'
        #x_sparse_path = name + '_input_example_sparse' + str(i) + '.png'
        #x_color_path = name + '_input_example_color' + str(i) + '.png'
        #y_path = name + '_gt_example' + str(i) + '.png'
        
        img_depth_sparse = np.hstack([y_img, output_example_img])
        img_out_color = np.hstack([x_sparse_img, x_color_img])
        img_all = np.vstack([img_depth_sparse, img_out_color])
        cv2.imwrite(output_path, img_all)
        #cv2.imwrite(x_sparse_path, x_sparse_img)
        #cv2.imwrite(x_color_path, x_color_img)
        #cv2.imwrite(y_path, y_img)

        #if not log_output_example.logged:
        run.log_image(name= name + '_Example' + str(i),
                    path=output_path,
                    plot=None,
                    description= name + '_Example' + str(i))

        #run.log_image(name= name + '_Example of input sparse' + str(i),
        #            path=x_sparse_path,
        #            plot=None,
        #            description= name + '_Example of input sparse' + str(i))
        
        #run.log_image(name= name + '_Example of input color' + str(i),
        #            path=x_color_path,
        #            plot=None,
        #            description= name + '_Example of input color' + str(i))

        #run.log_image(name= name + '_Example of ground truth' + str(i),
        #            path=y_path,
        #            plot=None,
        #            description= name + '_Example of ground truth' + str(i))
    #log_output_example.logged = True

#log_output_example.logged = False