import torch
from torch import nn


class SparseConv(nn.Module):
    """
      Sparse convolution layer
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.bias = nn.Parameter(
            torch.rand([1, out_channels, 1, 1]),  # [batch_size, out_ch, k_h, k_w]
            requires_grad=True)

        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            padding=padding,
            stride=1
        )

        self.count_valid_entries = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.count_valid_entries.weight = nn.Parameter(
            data=torch.FloatTensor(torch.ones([1, 1, kernel_size, kernel_size])),  # [out_ch, in_ch, k_h, k_w]
            requires_grad=False)  # weights stay fixed

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, validity_mask):
        x = self.conv(validity_mask * x)
        normalizer = 1.0 / (self.count_valid_entries(validity_mask) + 1e-8)
        x = x * normalizer + self.bias
        x = self.relu(x)

        validity_mask = self.max_pool(validity_mask)

        return x, validity_mask


class SparseConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparse_layer1 = SparseConv(1, 16, kernel_size=11)
        self.sparse_layer2 = SparseConv(16, 16, kernel_size=7)
        self.sparse_layer3 = SparseConv(16, 16, kernel_size=5)
        self.sparse_layer4 = SparseConv(16, 16, kernel_size=3)
        self.sparse_layer5 = SparseConv(16, 16, kernel_size=3)
        self.sparse_layer6 = SparseConv(16, 1, kernel_size=1)

    def forward(self, x, validity_mask):
        x, validity_mask = self.sparse_layer1(x, validity_mask)
        x, validity_mask = self.sparse_layer2(x, validity_mask)
        x, validity_mask = self.sparse_layer3(x, validity_mask)
        x, validity_mask = self.sparse_layer4(x, validity_mask)
        x, validity_mask = self.sparse_layer5(x, validity_mask)
        x, validity_mask = self.sparse_layer6(x, validity_mask)

        return x
