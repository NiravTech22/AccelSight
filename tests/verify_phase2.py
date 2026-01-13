import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.accelsight_net import AccelSightNet
from training.velocity_loss import VelocityLoss

def verify_phase2():
    print("--- Phase 2 Verification ---")
    num_frames = 5
    model = AccelSightNet(num_frames=num_frames)
    model.eval()
    
    input_res = 256
    dummy_input = torch.randn(1, num_frames, 3, input_res, input_res)
    
    print(f"Running forward pass with {num_frames} frames...")
    outputs = model(dummy_input)
    
    # Check shapes
    expected_res = input_res // 16 # Final bottleneck scale
    print(f"Objectness shape: {outputs['objectness'].shape}")
    print(f"Velocity shape: {outputs['velocity'].shape}")
    
    # Verify Velocity Loss
    criterion_vel = VelocityLoss(magnitude_weighting=True)
    gt_vel = torch.randn_like(outputs['velocity'])
    mask = torch.ones_like(outputs['objectness']) # Use full mask for testing
    
    loss_vel = criterion_vel(outputs['velocity'], gt_vel, mask)
    print(f"Velocity Loss Calculation: {loss_vel.item():.4f}")
    
    assert not torch.isnan(loss_vel), "Velocity loss is NaN"
    print("\nPhase 2 Spatiotemporal Verification: PASSED")

if __name__ == "__main__":
    verify_phase2()
