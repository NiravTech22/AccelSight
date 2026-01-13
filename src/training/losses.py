import torch
import torch.nn as nn
import torch.nn.functional as F
from .velocity_loss import VelocityLoss

class AccelSightMultiTaskLoss(nn.Module):
    """
    Consolidated Multi-task Loss for AccelSight.
    Phased approach: Supports detection, motion, and re-id consistency.
    """
    def __init__(self, w_obj=1.0, w_bbox=5.0, w_vel=2.0, w_embed=1.0, w_control=10.0):
        super(AccelSightMultiTaskLoss, self).__init__()
        self.w_obj = w_obj
        self.w_bbox = w_bbox
        self.w_vel = w_vel
        self.w_embed = w_embed
        self.w_control = w_control
        
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.bbox_loss = nn.SmoothL1Loss()
        self.vel_loss = VelocityLoss(magnitude_weighting=True)
        self.control_loss = nn.HuberLoss()

    def contrastive_embedding_loss(self, embeddings, instance_ids):
        """
        Simple contrastive loss to pull same-ID embeddings together and push others apart.
        embeddings: (N, D) - Flattened object embeddings
        instance_ids: (N,) - Corresponding object IDs
        """
        if len(instance_ids) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) # (N, N)
        
        # Mask for same ID (positive pairs)
        pos_mask = (instance_ids.unsqueeze(0) == instance_ids.unsqueeze(1)).float()
        
        # We want similarity to be 1 for same ID, 0 for different
        # Simplified: MSE loss towards the mask
        return F.mse_loss(sim_matrix, pos_mask)

    def forward(self, preds, targets):
        """
        preds: dict containing objectness, bbox, velocity, embedding
        targets: dict containing gt_objectness, gt_bbox, gt_velocity, gt_ids
        """
        # 1. Objectness Loss
        loss_obj = self.obj_loss(preds["objectness"], targets["gt_objectness"])
        
        # Mask where objects are present
        mask = (targets["gt_objectness"] > 0.5)
        
        # 2. BBox and Velocity Loss (Only on objects)
        loss_bbox = torch.tensor(0.0, device=preds["bbox"].device)
        loss_vel = torch.tensor(0.0, device=preds["velocity"].device)
        loss_embed = torch.tensor(0.0, device=preds["embedding"].device)
        
        # 3. Control Loss (Global)
        loss_control = self.control_loss(preds["controls"], targets["gt_controls"])

        if mask.any():
            m_flat = mask.view(-1)
            # BBox Loss
            loss_bbox = self.bbox_loss(preds["bbox"].permute(0,2,3,1).reshape(-1, 4)[m_flat], 
                                     targets["gt_bbox"].permute(0,2,3,1).reshape(-1, 4)[m_flat])
            
            # Velocity Loss
            loss_vel = self.vel_loss(preds["velocity"], targets["gt_velocity"], mask.float())
            
            # Embedding Loss (Phase 3)
            # Extract object embeddings and IDs
            obj_embeddings = preds["embedding"].permute(0,2,3,1).reshape(-1, preds["embedding"].shape[1])[m_flat]
            obj_ids = targets["gt_ids"].view(-1)[m_flat]
            loss_embed = self.contrastive_embedding_loss(obj_embeddings, obj_ids)

        total_loss = (self.w_obj * loss_obj + 
                      self.w_bbox * loss_bbox + 
                      self.w_vel * loss_vel + 
                      self.w_embed * loss_embed +
                      self.w_control * loss_control)
        
        return {
            "total_loss": total_loss,
            "obj_loss": loss_obj,
            "bbox_loss": loss_bbox,
            "vel_loss": loss_vel,
            "embed_loss": loss_embed,
            "control_loss": loss_control
        }
