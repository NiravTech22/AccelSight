import torch
import torch.nn as nn


class ControlHead(nn.Module):
    """
    Stage F: Learning-Based Control Head.
    Processes global features to predict control setpoints (e.g., Steering, Throttle).
    Uses Global Average Pooling to aggregate spatial information for decision making.
    """
    def __init__(self, in_channels, control_dim=3, hidden_dim=256):
        super(ControlHead, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, control_dim),
            nn.Tanh()  # Assuming normalized control outputs in [-1, 1]
        )

    def forward(self, x):
        """
        Input: (B, C, H, W) features from the bottleneck.
        Output: (B, control_dim) control signals.
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.mlp(x)


if __name__ == "__main__":
    # Test
    head = ControlHead(in_channels=256, control_dim=4)
    dummy_feat = torch.randn(2, 256, 16, 16)
    controls = head(dummy_feat)
    print(f"Control Head output shape: {controls.shape}")  # Expected: (2, 4)
