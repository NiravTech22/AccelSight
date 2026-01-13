import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.accelsight_net import AccelSightNet
from training.losses import AccelSightMultiTaskLoss

def verify_phase5():
    print("--- Phase 5 Verification: Learning-Based Controls ---")
    control_dim = 4 # e.g., Throttle, Brake, Steering, Flaps
    model = AccelSightNet(num_frames=5, control_dim=control_dim)
    criterion = AccelSightMultiTaskLoss()
    
    # Input
    dummy_input = torch.randn(2, 5, 3, 256, 256)
    outputs = model(dummy_input)
    
    # Targets
    targets = {
        "gt_objectness": torch.zeros(2, 1, 16, 16),
        "gt_bbox": torch.randn(2, 4, 16, 16),
        "gt_velocity": torch.randn(2, 3, 16, 16),
        "gt_ids": torch.zeros(2, 16, 16),
        "gt_controls": torch.randn(2, control_dim) # Expert controls
    }
    
    losses = criterion(outputs, targets)
    
    print(f"Control Prediction Shape: {outputs['controls'].shape}")
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Control Loss: {losses['control_loss'].item():.4f}")
    
    assert outputs['controls'].shape == (2, control_dim), "Control prediction shape mismatch"
    assert losses['control_loss'] >= 0, "Control loss should be non-negative"
    print("\nPhase 5 Control Head Verification: PASSED")

if __name__ == "__main__":
    verify_phase5()
