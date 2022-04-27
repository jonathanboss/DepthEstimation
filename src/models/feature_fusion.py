import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FusionNetwork(nn.Module):
    def __init__(
            self,
            rgb_in_channels=3,
            sparse_in_channels=1,
            out_channels=1,
            features=[8, 16, 32, 64, 128],
    ):
        super(FusionNetwork, self).__init__()
        self.decoder = nn.ModuleList()
        self.encoder_rgb = nn.ModuleList()
        self.encoder_sparse = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # RGB encoder
        for feature in features:
            self.encoder_rgb.append(DoubleConv(rgb_in_channels, feature))
            rgb_in_channels = feature

        # Sparse encoder
        for feature in features:
            self.encoder_sparse.append(DoubleConv(sparse_in_channels, feature))
            sparse_in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.decoder.append(DoubleConv(feature, feature))

        self.bottleneck = DoubleConv(features[-1] * 2, features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, rgb, sparse, mask):
        for layer in self.encoder_rgb:
            rgb = layer(rgb)
            rgb = self.pool(rgb)

        for layer in self.encoder_sparse:
            sparse = layer(sparse)
            sparse = self.pool(sparse)

        # Late fusion
        fusion = torch.cat((rgb, sparse), dim=1)
        fusion = self.bottleneck(fusion)

        for layer in self.decoder:
            fusion = layer(fusion)

        return self.final_conv(fusion)
