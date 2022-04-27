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


class YNET(nn.Module):
    def __init__(
            self,
            rgb_in_channels=3,
            sparse_in_channels=1,
            out_channels=1,
            features=[16, 32, 64, 128],
    ):
        super(YNET, self).__init__()
        self.decoder = nn.ModuleList()
        self.encoder_rgb = nn.ModuleList()
        self.encoder_sparse = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # RGB encoder
        for feature in features:
            self.encoder_rgb.append(DoubleConv(rgb_in_channels, feature // 2))
            rgb_in_channels = feature // 2

        # Sparse encoder
        for feature in features:
            self.encoder_sparse.append(DoubleConv(sparse_in_channels, feature // 2))
            sparse_in_channels = feature // 2

        # Decoder
        for idx, feature in enumerate(reversed(features)):
            size_multiplier = 2
            if idx == 0:
                size_multiplier = 2

            self.decoder.append(
                nn.ConvTranspose2d(
                    feature * size_multiplier, feature, kernel_size=2, stride=2,
                )
            )
            self.decoder.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, rgb, sparse, mask):
        skip_connections_rgb = []
        skip_connections_sparse = []

        for layer in self.encoder_rgb:
            rgb = layer(rgb)
            skip_connections_rgb.append(rgb)
            rgb = self.pool(rgb)

        for layer in self.encoder_sparse:
            sparse = layer(sparse)
            skip_connections_sparse.append(sparse)
            sparse = self.pool(sparse)

        skip_connections_rgb = skip_connections_rgb[::-1]
        skip_connections_sparse = skip_connections_sparse[::-1]

        fusion = torch.cat((rgb, sparse), dim=1)
        fusion = self.bottleneck(fusion)

        for idx in range(0, len(self.decoder), 2):
            fusion = self.decoder[idx](fusion)
            fusion = torch.cat((fusion, torch.cat((skip_connections_rgb[idx//2], skip_connections_sparse[idx//2]), dim=1)), dim=1)
            fusion = self.decoder[idx+1](fusion)

        return self.final_conv(fusion)
