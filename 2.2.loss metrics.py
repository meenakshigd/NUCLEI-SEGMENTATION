"""
Stage 2 — Loss and metrics for the baseline U-Net.

Shared by later stages too (boundary head reuses the same BCE+Dice; the
Dice/IoU metric functions are what Stage 7's evaluation harness will call
for pixel-level scoring across all three model variants).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, eps=1e-6):
    """Soft Dice loss. logits: (B,1,H,W) raw, target: (B,1,H,W) in {0,1}."""
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    target = target.reshape(target.size(0), -1).float()

    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Used for both the nucleus head (Stage 2) and boundary head (Stage 3)."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        target = target.float()
        bce_loss = self.bce(logits, target)
        d_loss = dice_loss(logits, target)
        return self.bce_weight * bce_loss + self.dice_weight * d_loss


@torch.no_grad()
def dice_score(logits, target, threshold=0.5, eps=1e-6):
    """Hard Dice at a fixed threshold, for reporting (not for backprop)."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    target = target.float()

    preds = preds.reshape(preds.size(0), -1)
    target = target.reshape(target.size(0), -1)

    intersection = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


@torch.no_grad()
def iou_score(logits, target, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    target = target.float()

    preds = preds.reshape(preds.size(0), -1)
    target = target.reshape(target.size(0), -1)

    intersection = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 64, 64)
    target = (torch.rand(4, 1, 64, 64) > 0.7).float()

    loss_fn = BCEDiceLoss()
    loss = loss_fn(logits, target)
    print("BCEDiceLoss:", loss.item())

    print("Dice score:", dice_score(logits, target))
    print("IoU score:", iou_score(logits, target))

    # sanity: perfect prediction should give dice/iou close to 1
    perfect_logits = (target * 20) - 10  # saturates sigmoid to match target
    print("Dice on perfect prediction:", dice_score(perfect_logits, target))
    print("IoU on perfect prediction:", iou_score(perfect_logits, target))