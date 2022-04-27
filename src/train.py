import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch
from torch import masked_fill, nn
from torch.utils.data import Dataset, DataLoader
from azureml.core import Run
import argparse
import os
import logging

from data.dataset import DepthCompletionDataset
from data.matterport import MatterportDataset
from models.sparsity_invariant_cnn import SparseConvolutionalNetwork
from models.unet import UNET, Late_fusion_UNET
from utils import utils
from utils.metrics import Metrics

#TODO : augmentations
#TODO : Learning rate scheduling? https://stats.stackexchange.com/questions/324896/training-loss-increases-with-time
#TODO : artifacts? https://distill.pub/2016/deconv-checkerboard/

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
run = Run.get_context()

model_dict = {
    "SparseConvCNN": SparseConvolutionalNetwork(),
    "unet": UNET(),
    "unet_late_fusion": Late_fusion_UNET()
}


def one_epoch(model, data_loader, epoch_number, opt=None, batches_per_epoch = 1):
    device = next(model.parameters()).device
    train = False if opt is None else True
    model.train() if train else model.eval()
    losses, correct, total = [], 0, 0
    if not train:
        metrics = Metrics(device)
    

    for i, (x_sparse, x_color, y, validity_mask) in enumerate(data_loader):
        x_sparse, x_color, y, validity_mask = x_sparse.to(device, dtype=torch.float),x_color.to(device, dtype=torch.float), y.to(device, dtype=torch.float), validity_mask.to(device,
                                                                                                                 dtype=torch.float)
        #print("max", torch.max(x_color), "min", torch.min(x_color), "avg", torch.mean(x_color))
        
        with torch.set_grad_enabled(train):
            output = model(x_sparse, x_color)

        loss_function = nn.L1Loss() # TODO: implement custom loss function
        #unobserved_mask = torch.logical_and((y > 0).float(), (validity_mask == 0).float())
        masked_output = torch.mul(output, validity_mask)
        masked_gt =  torch.mul(y, validity_mask)
        loss = loss_function(masked_output, masked_gt)*output.numel()/validity_mask.sum()
        
        if not train:
            metrics.step(masked_output, masked_gt)
        
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()       

        losses.append(loss.item())
        
        if i >= batches_per_epoch:
            break     
        
    #print(result[0], result[1], result[2])
        
    if not train:
        utils.log_output_example(run, x_sparse, x_color, y, output, str(epoch_number))
        metrics.compute(run)
        metrics.reset()
        
    return np.mean(losses)


def train(model, loader_train, loader_valid, lr=1e-3, max_epochs=30, weight_decay=0., patience=15):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_accuracy = 0
    best_valid_accuracy_epoch = 0

    best_loss_epoch = 0
    best_loss = 10e10
    for epoch in range(max_epochs):
        print(f'--- Epoch {epoch + 1} / {max_epochs} ---')
        
        train_loss = one_epoch(model, loader_train, epoch, opt)
        train_losses.append(train_loss)

        valid_loss = one_epoch(model, loader_valid, epoch)
        valid_losses.append(valid_loss)

        print(f'Train loss: {train_loss}, Validation loss: {valid_loss}')
        run.log('Train loss', train_loss)
        run.log('Validation loss', valid_loss)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_loss_epoch = epoch
            
        if epoch > best_loss_epoch + patience:
            break            

    return train_losses, valid_losses


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

    # Create the dataloader
    batch_size = 8
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(data_valid, batch_size=batch_size, shuffle=True)
    print("Training batches :", len(train_dataloader), "Valid batches:", len(valid_dataloader))

    # Start training of model
    log.info(f"Cuda is available: {torch.cuda.is_available()}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_dict[model_name].to(device)
    train_losses, valid_losses = train(model, train_dataloader, valid_dataloader, max_epochs=100)#, weight_decay=0.01)
    plot_history(train_losses, valid_losses)
    # Generate an example output from the model
    x_sparse, x_color, y, mask = next(iter(valid_dataloader))
    x_sparse, x_color, y, mask = x_sparse.to(device, dtype=torch.float), x_color.to(device, dtype=torch.float), y.to(device, dtype=torch.float), mask.to(device, dtype=torch.float)
    output_example = model(x_sparse, x_color)
    
    utils.log_output_example(run, x_sparse, x_color, y, output_example, 'final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='datasets/matterport_undistorted2/',
        help='Path to the training data'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='unet_late_fusion',
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
