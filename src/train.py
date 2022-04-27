import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import masked_fill, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from azureml.core import Run
import azureml.core
import argparse
import os
import logging

from data.dataset import DepthCompletionDataset
from data.matterport import MatterportDataset
from models.sparsity_invariant_cnn import SparseConvolutionalNetwork
from models.unet import UNET, Late_fusion_UNET
from models.si_unet import SI_UNET
from models.feature_fusion import FusionNetwork
from models.ynet import YNET
from utils import utils
from utils.metrics import Metrics

#TODO : augmentations
#TODO : Learning rate scheduling? https://stats.stackexchange.com/questions/324896/training-loss-increases-with-time
#TODO : artifacts? https://distill.pub/2016/deconv-checkerboard/

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
run = Run.get_context()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model_dict = {
    "SparseConvCNN": SparseConvolutionalNetwork(),
    "UNET": UNET(),
    "SI-UNET": SI_UNET(),
    "unet_late_fusion": Late_fusion_UNET(),
    "FusionNetwork": FusionNetwork(),
    "YNET": YNET()
}


def one_epoch(model, data_loader, epoch_number, opt=None):
    device = next(model.parameters()).device
    train = False if opt is None else True
    model.train() if train else model.eval()
    losses, correct, total = [], 0, 0

    metrics = Metrics(device)
    
    for rgb, sparse, validity_mask, dense in data_loader:
        rgb = rgb.to(device, dtype=torch.float)
        sparse = sparse.to(device, dtype=torch.float)
        validity_mask = validity_mask.to(device, dtype=torch.float)
        dense = dense.to(device, dtype=torch.float)

        with torch.set_grad_enabled(train):
            output = model(rgb, sparse, validity_mask)
            
        loss_function = nn.L1Loss()
        unobserved_mask = torch.mul((dense > 0), ~(validity_mask > 0))
        masked_output = torch.mul(output, unobserved_mask)
        masked_gt = torch.mul(dense, unobserved_mask)
        loss = loss_function(masked_output, masked_gt) * output.numel() / unobserved_mask.sum()
        
        if not train:
            metrics.step(masked_output, masked_gt)
        
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()       

        losses.append(loss.item())
        
    if not train:
        utils.log_output_example(run, sparse, rgb, dense, output, str(epoch_number))
        metrics.compute(run)
        metrics.reset()
        
    return np.mean(losses)


def train(model, loader_train, loader_valid, lr=1e-3, max_epochs=40, weight_decay=0., patience=10):
    train_losses = []
    valid_losses = []
    best_model = 0

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_loss = float('inf')
    best_valid_loss_epoch = 0

    best_loss_epoch = 0
    best_loss = 10e10
    for epoch in range(max_epochs):
        print(f'--- Epoch {epoch + 1} / {max_epochs} ---')

        train_loss = one_epoch(model, loader_train, epoch, opt)
        train_losses.append(train_loss)

        valid_loss = one_epoch(model, loader_valid, epoch)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_loss_epoch = epoch
            best_model = model.state_dict()

        if epoch > best_valid_loss_epoch + patience:
            log.info("Validation loss not getting any lower. Stopping.")
            break

        print(f'Train loss: {train_loss}, Validation loss: {valid_loss}')
        run.log('Train loss', train_loss)
        run.log('Validation loss', valid_loss)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_loss_epoch = epoch
            
        if epoch > best_loss_epoch + patience:
            break            

    save_model(run, best_model)
    return train_losses, valid_losses

def save_model(run, model):
    # save on local machine
    os.makedirs('./outputs', exist_ok=True)
    torch.save(model, 'outputs/model.pt')

    if isinstance(run, azureml.core.run._OfflineRun) is False:
        # Register the model
        run.register_model(model_name='depth-completion-model', model_path='outputs/model.pt')

        # Create a model folder in the current directory
        os.makedirs('./model', exist_ok=True)

        # Download the model from run history
        run.download_file(name='outputs/model.pt', output_file_path='./model/model.pt'),


def plot_history(train_losses, valid_losses):
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 1, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    p = plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    run.log_image(name='Metrics history',
                  path=None,
                  plot=plt,
                  description='Update history of model evaluation metrics')


def main(data_path, model_name):
    # Create dataset
    print("model name:", model_name)
    dataset = MatterportDataset(data_path)
    print("Dataset size :", len(dataset))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    data_train, data_valid = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Size of training dataset:", train_size)
    print("Size of validation dataset:", test_size)

    # Create the dataloader
    batch_size = 8
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(data_valid, batch_size=batch_size, shuffle=True)
    print("Training batches :", len(train_dataloader), "Valid batches:", len(valid_dataloader))

    # Start training of model
    log.info(f"Cuda is available: {torch.cuda.is_available()}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_dict[model_name].to(device)

    plot_history(*train(model, train_dataloader, valid_dataloader, max_epochs=100, weight_decay=1e-5))

    # Generate an example output from the model
    rgb, sparse, validity_mask, dense = next(iter(train_dataloader))
    rgb = rgb.to(device, dtype=torch.float)
    sparse = sparse.to(device, dtype=torch.float)
    validity_mask = validity_mask.to(device, dtype=torch.float)
    dense = dense.to(device, dtype=torch.float)

    output_example = model(rgb, sparse, validity_mask)

    utils.log_output_example(run, sparse, rgb, dense, output_example, 'final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='../datasets/matterport_undistorted2/',
        help='Path to the training data'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='YNET',
        help='Name of the model to train'
    )

    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.getcwd())
    print(os.listdir(args.data_path))
    print("================")

    main(args.data_path, args.model)
