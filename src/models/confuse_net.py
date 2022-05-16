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


class ConFuseNet(nn.Module):
    def __init__(
            self,
            rgb_in_channels=3,
            sparse_in_channels=1,
            out_channels=1,
            features=[16, 32, 64, 128],
    ):
        super(ConFuseNet, self).__init__()
        self.decoder = nn.ModuleList()
        self.encoder_rgb = nn.ModuleList()
        self.encoder_sparse = nn.ModuleList()
        self.encoder_fusion = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # RGB encoder
        for feature in features:
            self.encoder_rgb.append(DoubleConv(rgb_in_channels, feature // 2))
            rgb_in_channels = feature // 2

        # Sparse encoder
        for feature in features:
            self.encoder_sparse.append(DoubleConv(sparse_in_channels, feature // 2))
            sparse_in_channels = feature // 2

        # Fusion encoder
        features_in = features[0]
        self.encoder_fusion.append(DoubleConv(features_in, features[1]))
        for feature in features[1:]:
            self.encoder_fusion.append(DoubleConv(feature*2, feature*2))

        # Decoder
        for idx, feature in enumerate(reversed(features)):
            self.decoder.append(
                nn.ConvTranspose2d(
                    feature * 4, feature * 2, kernel_size=2, stride=2,
                )
            )
            self.decoder.append(DoubleConv(feature * 4, feature * 2))

        self.bottleneck = DoubleConv(features[-1] * 2, features[-1] * 4)
        self.final_conv = nn.Conv2d(features[0] * 2, out_channels, kernel_size=1)

    def forward(self, rgb, sparse, mask):
        skip_connections_rgb = []
        skip_connections_sparse = []
        skip_connections_fusion = []

        for layer in self.encoder_rgb:
            rgb = layer(rgb)
            skip_connections_rgb.append(rgb)
            rgb = self.pool(rgb)

        for layer in self.encoder_sparse:
            sparse = layer(sparse)
            skip_connections_sparse.append(sparse)
            sparse = self.pool(sparse)

        skip_connections_rgb = skip_connections_rgb
        skip_connections_sparse = skip_connections_sparse

        # Fusion decoder
        fusion = torch.cat((skip_connections_rgb[0], skip_connections_sparse[0]), dim=1)
        for idx, layer in enumerate(self.encoder_fusion):
            fusion = layer(fusion)
            skip_connections_fusion.append(fusion)
            fusion = self.pool(fusion)
            if idx != len(self.encoder_fusion) - 1:
                fusion = torch.cat((fusion, torch.cat((skip_connections_rgb[idx + 1], skip_connections_sparse[idx + 1]), dim=1)), dim=1)
        skip_connections_fusion = skip_connections_fusion[::-1]

        fusion = self.bottleneck(fusion)

        for idx in range(0, len(self.decoder), 2):
            fusion = self.decoder[idx](fusion)
            fusion = torch.cat((fusion, skip_connections_fusion[idx // 2]), dim=1)
            fusion = self.decoder[idx+1](fusion)

        return self.final_conv(fusion)
