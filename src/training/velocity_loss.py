import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityLoss(nn.Module):
    """
    Computes loss for velocity vector regression.
    Optionally weights the loss by the magnitude of the ground truth velocity.
    """
    def __init__(self, magnitude_weighting=False):
        super(VelocityLoss, self).__init__()
        self.magnitude_weighting = magnitude_weighting
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_vel, gt_vel, mask):
        """
        pred_vel: (B, 3, H, W)
        gt_vel: (B, 3, H, W)
        mask: (B, 1, H, W) - Boolean mask where objects exist
        """
        if not mask.any():
            return torch.tensor(0.0, device=pred_vel.device, requires_grad=True)

        # Apply mask
        loss = self.criterion(pred_vel, gt_vel)
        loss = loss * mask.expand_as(loss)
        
        if self.magnitude_weighting:
            # Increase importance of error for fast moving objects
            mag = torch.norm(gt_vel, p=2, dim=1, keepdim=True)
            weight = 1.0 + mag
            loss = loss * weight.expand_as(loss)

        return loss.sum() / mask.sum()
