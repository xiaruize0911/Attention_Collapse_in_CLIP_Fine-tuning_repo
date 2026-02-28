"""
regularizer.py - Attention Preservation Regularizer (APR) and Entropy Floor
"""
import torch
import torch.nn as nn


class AttentionPreservationRegularizer:
    """
    APR: Constrains fine-tuned attention to stay close to pretrained attention
    using KL divergence.
    """
    
    def __init__(self, pretrained_vision_model, lambda_reg=0.01, layers=None, device='cuda'):
        self.lambda_reg = lambda_reg
        self.layers = layers or list(range(12))
        self.device = device
        
        # Keep pretrained model frozen for reference
        self.pretrained_model = pretrained_vision_model.to(device).eval()
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
    
    def compute(self, pixel_values, current_attentions):
        """
        Compute APR loss.
        pixel_values: (B, 3, 224, 224)
        current_attentions: tuple of (B, H, seq, seq) from current model
        Returns: scalar loss
        """
        with torch.no_grad():
            pretrained_outputs = self.pretrained_model(
                pixel_values=pixel_values,
                output_attentions=True
            )
            pretrained_attentions = pretrained_outputs.attentions
        
        kl_loss = 0.0
        for l in self.layers:
            # CLS -> patch attention
            curr = current_attentions[l][:, :, 0, 1:]  # (B, H, 49)
            pret = pretrained_attentions[l][:, :, 0, 1:]
            
            # Renormalize
            curr = curr / (curr.sum(dim=-1, keepdim=True) + 1e-10)
            pret = pret / (pret.sum(dim=-1, keepdim=True) + 1e-10)
            
            # KL(current || pretrained)
            kl = (curr * (torch.log(curr + 1e-10) - torch.log(pret + 1e-10))).sum(dim=-1)
            kl_loss += kl.mean()
        
        kl_loss /= len(self.layers)
        return self.lambda_reg * kl_loss


class EntropyFloorRegularizer:
    """
    Entropy Floor: Penalizes attention entropy below a threshold.
    """
    
    def __init__(self, baseline_entropy_per_layer, floor_ratio=0.7, lambda_reg=0.01):
        """
        baseline_entropy_per_layer: list of 12 floats (pretrained attention entropy per layer)
        floor_ratio: fraction of baseline entropy to use as floor (e.g., 0.7 = 70%)
        """
        self.lambda_reg = lambda_reg
        self.thresholds = [e * floor_ratio for e in baseline_entropy_per_layer]
    
    def compute(self, current_attentions):
        """
        Compute entropy floor loss.
        current_attentions: tuple of (B, H, seq, seq)
        """
        loss = 0.0
        for l, threshold in enumerate(self.thresholds):
            if l >= len(current_attentions):
                break
            cls_attn = current_attentions[l][:, :, 0, 1:]  # (B, H, 49)
            cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Compute entropy
            entropy = -(cls_attn * torch.log(cls_attn + 1e-10)).sum(dim=-1)  # (B, H)
            
            # Penalize when entropy < threshold
            penalty = torch.relu(threshold - entropy)  # (B, H)
            loss += penalty.mean()
        
        return self.lambda_reg * loss
