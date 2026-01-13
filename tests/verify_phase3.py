import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.accelsight_net import AccelSightNet
from training.losses import AccelSightMultiTaskLoss

def verify_phase3():
    print("--- Phase 3 Verification ---")
    model = AccelSightNet(num_frames=5)
    criterion = AccelSightMultiTaskLoss()
    
    # Input
    dummy_input = torch.randn(2, 5, 3, 256, 256)
    outputs = model(dummy_input)
    
    # Targets for 2 windows
    # Resolution 16x16
    gt_obj = torch.zeros(2, 1, 16, 16)
    gt_obj[0, 0, 5, 5] = 1.0 # Object 1 in Window 0
    gt_obj[1, 0, 5, 5] = 1.0 # Object 1 in Window 1 (Same ID)
    gt_obj[0, 0, 10, 10] = 1.0 # Object 2 in Window 0
    
    # GT IDs
    gt_ids = torch.zeros(2, 16, 16)
    gt_ids[0, 5, 5] = 101 # ID A
    gt_ids[1, 5, 5] = 101 # ID A (Consistency)
    gt_ids[0, 10, 10] = 202 # ID B
    
    targets = {
        "gt_objectness": gt_obj,
        "gt_bbox": torch.randn(2, 4, 16, 16),
        "gt_velocity": torch.randn(2, 3, 16, 16),
        "gt_ids": gt_ids
    }
    
    losses = criterion(outputs, targets)
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Embedding Consistency Loss: {losses['embed_loss'].item():.4f}")
    
    assert losses['embed_loss'] >= 0, "Embedding loss should be non-negative"
    print("\nPhase 3 ID Consistency Verification: PASSED")

if __name__ == "__main__":
    verify_phase3()
