import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Deeply modular residual block for spatial encoding."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SpatialEncoder(nn.Module):
    """
    Stage A: Modular Spatial Encoder.
    Progressively extracts spatial features and reduces resolution.
    """
    def __init__(self, in_channels=3, base_channels=16):
        super(SpatialEncoder, self).__init__()

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Layers: Stride 2 downsamples resolution by half each time
        self.layer1 = ResidualBlock(base_channels, base_channels * 2, stride=2)  # 1/2 resolution
        self.layer2 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)  # 1/4 resolution
        self.layer3 = ResidualBlock(base_channels * 4, base_channels * 8, stride=2)  # 1/8 resolution

        self.out_channels = base_channels * 8

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


if __name__ == "__main__":
    # Quick shape check
    encoder = SpatialEncoder()
    dummy = torch.randn(1, 3, 256, 256)
    out = encoder(dummy)
    print(f"SpatialEncoder output shape: {out.shape}")  # Expected: (1, 128, 32, 32)
