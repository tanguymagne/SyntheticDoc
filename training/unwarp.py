import torch
import torch.nn.functional as F


def bilinear_unwarping(warped_img, grid_2d, img_size):
    """Unwarp warped_img (BxCxHxW) using grid_2d (Bx2xGhxGw) at img_size (w, h)."""
    upsampled_grid = F.interpolate(
        grid_2d, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(
        warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True
    )

    return unwarped_img


def bilinear_unwarping_from_numpy(warped_img, grid_2d, img_size):
    """Same as bilinear_unwarping, but for HxWxC numpy arrays."""
    warped_img = torch.unsqueeze(torch.from_numpy(warped_img.transpose(2, 0, 1)).float(), dim=0)
    grid_2d = torch.unsqueeze(torch.from_numpy(grid_2d.transpose(2, 0, 1)).float(), dim=0)

    unwarped_img = bilinear_unwarping(warped_img, grid_2d, img_size)

    unwarped_img = unwarped_img[0].numpy().transpose(1, 2, 0)
    return unwarped_img
