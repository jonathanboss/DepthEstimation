from data.matterport import MatterportDataset
from models.ynet import YNET
from torch.utils.data import Dataset, DataLoader
import logging
import os
import torch
from torch import masked_fill, nn
from utils.metrics import Metrics
from utils import utils
from azureml.core import Run
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import numpy as np
from scipy.interpolate import griddata

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
run = Run.get_context()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def upsample(img_sparse, method):
    img_sparse = img_sparse.numpy().squeeze()
    grid_x, grid_y = np.mgrid[0:img_sparse.shape[0], 0:img_sparse.shape[1]]
    points = np.array(np.where(img_sparse > 0.0))
    values = img_sparse[np.where(img_sparse > 0.0)]
    dense_interpolated = griddata(points.T, values, (grid_x, grid_y), method=method)
    dense_interpolated = np.nan_to_num(dense_interpolated, nan=0.0)
    return torch.from_numpy(dense_interpolated).unsqueeze(0).unsqueeze(0)

class Metrics():
    def __init__(self, device, metrics=None):
        if metrics is None:
            metrics = {'MAE': MeanAbsoluteError(),
                       'RMSE': MeanSquaredError(squared=False),
                       'MSE': MeanSquaredError(squared=True)}
        self.metrics = metrics
        for name, metric in self.metrics.items():
            metric.to(device)
        self.mae_all = []
        self.rmse_all = []
        self.mse_all = []


    def step(self, prediction, target):
        for name, metric in self.metrics.items():
            metric(prediction, target)

    def compute(self):
        for name, metric in self.metrics.items():
            value = metric.compute().item()
            print(name, value)


    def reset(self):
        for name, metric in self.metrics.items():
            metric.reset()

def evaluate(model, data_loader, device):
    model.eval()
    losses = []

    metrics_dnn = Metrics(device)
    metrics_upsample = Metrics(device)

    idx = 1
    for rgb, sparse, validity_mask, dense in data_loader:
        if (idx % 250 == 0):
            print(f'----- {idx} / {len(data_loader)} -----')
        rgb = rgb.to(device, dtype=torch.float)
        sparse = sparse.to(device, dtype=torch.float)
        validity_mask = validity_mask.to(device, dtype=torch.float)
        dense = dense.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):
            output = model(rgb, sparse, validity_mask)

        loss_function = nn.L1Loss()
        unobserved_mask = dense > 0
        masked_output = torch.mul(output, unobserved_mask)
        masked_gt = torch.mul(dense, unobserved_mask)
        loss = loss_function(masked_output, masked_gt) * output.numel() / unobserved_mask.sum()

        metrics_dnn.step((masked_output * output.numel() / unobserved_mask.sum()) * 16.384,  (masked_gt * output.numel() / unobserved_mask.sum()) * 16.384)

        validity_idx = torch.nonzero(dense, as_tuple=True)
        observations_mask = torch.zeros(dense.shape)
        observations_mask[validity_idx] = 1

        losses.append(loss.item())

        #name = "evaluation_" + str(idx)
        #utils.log_output_example(run, sparse, rgb, dense, output, name)
        idx += 1

    print("----- YNET ----- ")
    metrics_dnn.compute()
    metrics_dnn.reset()
    print("----- Upsampling ----- ")
    metrics_upsample.compute()
    metrics_upsample.reset()


def main(data_path, model_path):
    sparsity_levels = [0.01, 0.005, 0.001]
    log.info(f"Cuda is available: {torch.cuda.is_available()}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for name, path in model_path.items():
        print(f"------------- Model: {name} -------------")
        for level in sparsity_levels:
            print(f"* Sparsity level: {level} *")
            dataset = MatterportDataset(data_path, level)

            batch_size = 1
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            model = YNET().to(device)

            model.load_state_dict((torch.load(path, map_location=device)))
            evaluate(model, dataloader, device)


if __name__ == '__main__':
    data_path = '../datasets/matterport_undistorted2/'
    model_paths = {"feature": 'outputs/model_features.pt',
                   "uniform": 'outputs/model_features.pt'}
    main(data_path, model_paths)
