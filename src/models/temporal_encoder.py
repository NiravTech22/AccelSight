import torch
import torch.nn as nn


class SpatioTemporalEncoder(nn.Module):
    """
    Stage B: Temporal Encoding.
    Accepts a 5D tensor (Batch, Channels, Depth/Time, Height, Width)
    and applies 3D convolutions to capture motion patterns.
    """
    def __init__(self, in_channels, out_channels, time_kernel=3):
        super(SpatioTemporalEncoder, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(time_kernel, 3, 3),
                      padding=(time_kernel // 2, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        """
        Input: (B, C, T, H, W)
        Output: (B, C_out, T, H, W)
        """
        return self.conv3d(x)


if __name__ == "__main__":
    # Internal test
    encoder = SpatioTemporalEncoder(128, 256)
    dummy = torch.randn(1, 128, 5, 32, 32)
    out = encoder(dummy)
    print(f"SpatioTemporalEncoder output shape: {out.shape}")
