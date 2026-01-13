import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.spatial_encoder import SpatialEncoder
from models.lite_head import LiteDetectionHead

class Phase1Model(nn.Module):
    def __init__(self):
        super(Phase1Model, self).__init__()
        self.encoder = SpatialEncoder()
        self.det_head = LiteDetectionHead(self.encoder.out_channels)

    def forward(self, x):
        feat = self.encoder(x)
        obj, bbox = self.det_head(feat)
        return obj, bbox

def verify_phase1():
    print("--- Phase 1 Verification ---")
    model = Phase1Model()
    input_res = 256
    dummy_input = torch.randn(2, 3, input_res, input_res)
    
    obj, bbox = model(dummy_input)
    
    expected_res = input_res // 8
    print(f"Input shape: {dummy_input.shape}")
    print(f"Objectness shape: {obj.shape} (Expected: (2, 1, {expected_res}, {expected_res}))")
    print(f"BBox shape: {bbox.shape} (Expected: (2, 4, {expected_res}, {expected_res}))")
    
    assert obj.shape == (2, 1, expected_res, expected_res), "Objectness shape mismatch"
    assert bbox.shape == (2, 4, expected_res, expected_res), "BBox shape mismatch"
    print("\nPhase 1 Architecture Shape Verification: PASSED")

if __name__ == "__main__":
    verify_phase1()
