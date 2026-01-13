import torch.nn as nn


class PredictionHead(nn.Module):
    """General purpose 1x1 conv prediction head."""
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super(PredictionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)


class LiteDetectionHead(nn.Module):
    """
    Combined Objectness and BBox head.
    Splits output into confidence map and coordinate offsets.
    """
    def __init__(self, in_channels):
        super(LiteDetectionHead, self).__init__()
        # 1 channel for objectness, 4 for (x, y, w, h)
        self.head = PredictionHead(in_channels, 1 + 4)

    def forward(self, x):
        out = self.head(x)
        objectness = out[:, :1, :, :]
        bbox = out[:, 1:, :, ...]
        return objectness, bbox


class LiteRegressionHead(nn.Module):
    """Head for Velocity or other vector regression tasks."""
    def __init__(self, in_channels, out_dim=3):
        super(LiteRegressionHead, self).__init__()
        self.head = PredictionHead(in_channels, out_dim)

    def forward(self, x):
        return self.head(x)
