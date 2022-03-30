import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from azureml.core import Run
import argparse
import os
import logging
import cv2
import torchvision

from data.dataset import DepthCompletionDataset
from data.matterport import MatterportDataset
from models.sparsity_invariant_cnn import SparseConvolutionalNetwork
from utils.utils import make_depth_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
run = Run.get_context()


def one_epoch(model, data_loader, opt=None):
    device = next(model.parameters()).device
    train = False if opt is None else True
    model.train() if train else model.eval()
    losses, correct, total = [], 0, 0
    for x, y, mask in data_loader:
        x, y, mask = x.to(device, dtype=torch.float), y.to(device, dtype=torch.float), mask.to(device,
                                                                                               dtype=torch.float)
        with torch.set_grad_enabled(train):
            output = model(x, mask)

        loss_function = nn.MSELoss()  # TODO: implement custom loss function
        loss = (loss_function(output, y) * mask.detach()).sum() / mask.sum()

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append(loss.item())
    return np.mean(losses)


def train(model, loader_train, loader_valid, lr=1e-3, max_epochs=30, weight_decay=0., patience=5):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_accuracy = 0
    best_valid_accuracy_epoch = 0

    for epoch in range(max_epochs):
        print(f'--- Epoch {epoch+1} / {max_epochs} ---')
        train_loss = one_epoch(model, loader_train, opt)
        train_losses.append(train_loss)

        valid_loss = one_epoch(model, loader_valid)
        valid_losses.append(valid_loss)

        print(f'Train loss: {train_loss}, Validation loss: {valid_loss}')
        run.log('Train loss', train_loss)
        run.log('Validation loss', valid_loss)

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


def main(data_path):
    # Create dataset
    dataset = MatterportDataset(data_path)

    print('Size of all data:', len(dataset))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    data_train, data_valid = torch.utils.data.random_split(dataset, [train_size, test_size])

    print('Size of train data:', len(data_train))
    print('Size of validation data:', len(data_valid))

    # Create the dataloader
    batch_size = 2
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)

    # Start training of model
    log.info(f"Cuda is available: {torch.cuda.is_available()}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SparseConvolutionalNetwork().to(device)
    plot_history(*train(model, train_dataloader, valid_dataloader, max_epochs=40, weight_decay=0.01))

    # Generate an example output from the model
    x, y, mask = next(iter(valid_dataloader))
    x, y, mask = x.to(device, dtype=torch.float), y.to(device, dtype=torch.float), mask.to(device, dtype=torch.float)

    output_example = model(x, mask)
    output_example = output_example[0].cpu().detach().numpy()
    output_example = make_depth_image(output_example)

    x = x[0].cpu().detach().numpy()
    y = y[0].cpu().detach().numpy()
    x = make_depth_image(x)
    y = make_depth_image(y)

    cv2.imwrite('output_example.png', output_example)
    cv2.imwrite('input_example.png', x)
    cv2.imwrite('gt_example.png', y)

    run.log_image(name='Example of output',
                  path='output_example.png',
                  plot=None,
                  description='Example of output')

    run.log_image(name='Example of input',
                  path='input_example.png',
                  plot=None,
                  description='Example of input')

    run.log_image(name='Example of ground truth',
                  path='gt_example.png',
                  plot=None,
                  description='Example of ground truth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='datasets/matterport/',
        help='Path to the training data'
    )

    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.getcwd())
    print(os.listdir(args.data_path))
    print("================")

    main(args.data_path)
