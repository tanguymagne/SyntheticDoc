import torch
import torch.nn as nn
import torch.nn.functional as F
from unwarp import bilinear_unwarping


def _reconstruction_loss(inp, bm_pred, bm_gt, reg_loss):
    """Compare the images unwarped with the predicted and the ground-truth backward maps."""
    B, C, H, W = inp.shape
    inp_01 = inp * 0.5 + 0.5  # [-1, 1] -> [0, 1]

    bm_pred_up = F.interpolate(bm_pred.float(), size=(H, W), mode="bilinear", align_corners=True)
    bm_gt_up = F.interpolate(bm_gt.float(), size=(H, W), mode="bilinear", align_corners=True)

    recon_pred = bilinear_unwarping(inp_01, bm_pred_up, img_size=(W, H))
    recon_gt = bilinear_unwarping(inp_01, bm_gt_up, img_size=(W, H))

    return reg_loss(recon_pred, recon_gt)


class MultiObjectiveLoss(nn.Module):
    """Weighted sum of the backward map, shadow map and (optional) reconstruction losses."""

    def __init__(self, bm_weight=1.0, shadow_weight=0.5, recon_weight=0.0, loss_type="l1"):
        super().__init__()
        self.bm_weight = bm_weight
        self.shadow_weight = shadow_weight
        self.recon_weight = recon_weight
        self.reg_loss = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()

    def forward(self, predictions, targets):
        """Return the total loss along with each individual term, for logging."""
        bm_pred, shadow_pred = predictions["bm_map"], predictions["shadow_map"]
        bm_gt, shadow_gt = targets["bm_gt"], targets["shadow_gt"]

        bm_loss = self.reg_loss(bm_pred, bm_gt)
        shadow_loss = self.reg_loss(shadow_pred, shadow_gt)

        total = self.bm_weight * bm_loss + self.shadow_weight * shadow_loss

        recon_loss = torch.tensor(0.0, device=bm_pred.device)
        if self.recon_weight > 0.0:
            recon_loss = _reconstruction_loss(targets["input"], bm_pred, bm_gt, self.reg_loss)
            total = total + self.recon_weight * recon_loss

        return {
            "total_loss": total,
            "bm_loss": bm_loss,
            "shadow_loss": shadow_loss,
            "recon_loss": recon_loss,
        }
