import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


#============================================================================
# Augmentation techniques
#============================================================================

def adjust_brightness_contrast(patch, brightness, contrast):
    return ((patch - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)


def gaussian_blur_patch(patch, ksize):
    if ksize <= 1:
        return patch
    elif ksize % 2 == 0:
        ksize += 1

    return F.avg_pool2d(patch, kernel_size=ksize, stride=1, padding=ksize // 2)


def paste_rotated_patch(search, patch, x_pos, y_pos, angle_deg):
    """
    Paste a rotated patch without clipping its corners.

    Args:
        search:         (1,3,H,W)
        patch:          (1,3,PH,PW)
        x_pos,y_pos:    top-left position of the original unrotated patch
        angle_deg:      rotation angle in degrees
    """

    device = search.device
    dtype = search.dtype

    _, _, H, W = search.shape
    _, _, PH, PW = patch.shape

    # Larger canvas so rotated patch corners are not clipped
    diagonal = int(np.ceil(np.sqrt(PH ** 2 + PW ** 2)))                             # The largest size a rotated patch can occupy is its diagonal
    canvas = torch.zeros((1, 3, diagonal, diagonal), device=device, dtype=dtype)
    mask = torch.zeros((1, 1, diagonal, diagonal), device=device, dtype=dtype)      # binary mask for patch (1 = patch pixel, 0 = search image)

    y_pad = (diagonal - PH) // 2
    x_pad = (diagonal - PW) // 2

    # Paste the patch into the search image
    canvas[:, :, y_pad:y_pad + PH, x_pad:x_pad + PW] = patch
    mask[:, :, y_pad:y_pad + PH, x_pad:x_pad + PW] = 1.0

    # Counter-clockwise rotation matrix
    theta = np.deg2rad(angle_deg)
    rot = torch.tensor(
        [[[np.cos(theta), -np.sin(theta), 0.0],
          [np.sin(theta),  np.cos(theta), 0.0]]],
        dtype=dtype,
        device=device,
    )

    # Rotate the patch
    grid = F.affine_grid(rot, canvas.size(), align_corners=False)
    canvas_rotated = F.grid_sample(canvas, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask_rotated = F.grid_sample(mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask_rotated = (mask_rotated > 0.5).float()

    # Compute patch center & top-left corner of the larger canvas
    center_x = x_pos + PW // 2
    center_y = y_pos + PH // 2
    new_x_pos = int(center_x - diagonal // 2)
    new_y_pos = int(center_y - diagonal // 2)

    # Clip to image bounds
    x_start = max(0, new_x_pos)
    y_start = max(0, new_y_pos)
    x_end = max(0, min(W, new_x_pos + diagonal))
    y_end = max(0, min(H, new_y_pos + diagonal))

    # Define the region of the canvas to copy such that it exactly matches the visible region of the search image.
    # cx_end/cy_end can go negative when the patch sits entirely past the right/bottom edge (x_start/y_start
    # already beyond W/H, so x_end/y_end get clamped down to W/H while new_x_pos/new_y_pos stay far larger) -
    # clamp to 0 there too, or a negative Python slice end silently wraps around instead of giving an empty slice.
    cx_start = x_start - new_x_pos
    cy_start = y_start - new_y_pos
    cx_end = max(0, cx_start + (x_end - x_start))
    cy_end = max(0, cy_start + (y_end - y_start))

    x_adv = search.clone()

    # Extract region of interest and patch
    roi = x_adv[:, :, y_start : y_end, x_start : x_end]
    patch_roi = canvas_rotated[:, :, cy_start : cy_end, cx_start : cx_end]
    mask_roi = mask_rotated[:, :, cy_start : cy_end, cx_start : cx_end]

    # Paste rotated patch
    roi = roi * (1 - mask_roi) + patch_roi * mask_roi
    x_adv[:, :, y_start : y_end, x_start : x_end] = roi

    return x_adv


# Paste the adversarial patch into the ORIGINAL (full-resolution) frame instead of the small
# cropped/resized search image the tracker actually sees, for visualization purposes.
def paste_patch_on_frame(frame, patch, crop_center_xy, crop_s_x, out_sz, patch_x, patch_y, patch_rotation, patch_size, device):

    frame_tensor = torch.from_numpy(frame).to(device).permute(2, 0, 1).unsqueeze(0).float()

    patch_for_frame = patch.to(device=device, dtype=frame_tensor.dtype)
    if frame_tensor.max() > 1.0 and patch_for_frame.max() <= 1.0:
        patch_for_frame = patch_for_frame * 255.0

    # RGB -> BGR, to match the frame's channel order
    patch_for_frame = patch_for_frame[:, [2, 1, 0], :, :]

    patch_resized = F.interpolate(patch_for_frame, size=(patch_size, patch_size), mode="bilinear", align_corners=False)

    # Map the local (search-image) top-left position into original-frame coordinates
    scale = crop_s_x / out_sz
    global_x = int(round(crop_center_xy[0] - crop_s_x / 2 + patch_x * scale))
    global_y = int(round(crop_center_xy[1] - crop_s_x / 2 + patch_y * scale))

    frame_patched = paste_rotated_patch(frame_tensor, patch_resized, global_x, global_y, patch_rotation)

    frame_patched = frame_patched[0].detach().cpu().permute(1, 2, 0).numpy()
    frame_patched = np.clip(frame_patched, 0, 255).astype(np.uint8)
    return frame_patched


def load_patch(checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    patch = checkpoint["patch"] # (1,3,H,W)

    patch = patch.detach().float().to(device)
    return patch.clamp(0.0, 1.0)


# Concatenate all frames to build mp4 video
def make_video(image_dir, output_file, fps=30):

    # file paths of all .jpg images
    images_all_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not images_all_paths:
        return

    first = cv2.imread(images_all_paths[0])
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),)

    # Iterate over all frames and add them to the video
    for img_path in images_all_paths:
        frame = cv2.imread(img_path)
        writer.write(frame)

    writer.release()


# Save tensor with shape (1,3,H,W) or (3,H,W) as an image
def save_tensor_as_image(image_tensor: torch.Tensor, path: Path):

    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]

    img = image_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).round().astype(np.uint8)

    succeed = cv2.imwrite( str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not succeed:
        raise RuntimeError(f"Saving image failed.")


# Return foreground probability for each of the anchors at every response map cell
def foreground_scores(out_cls):

    batch, channels, height, width = out_cls.shape

    num_anchors = channels // 2
    logits = out_cls.reshape(batch, 2, num_anchors, height, width)

    prob = F.softmax(logits, dim=1)

    foreground_prob = prob[:, 1] # 0 = background, 1 = foreground
    return foreground_prob


# Build all 19x19x5 anchors of the SiamRPN tracker
def build_anchors(response_sz=19, total_stride=8, ratios=(0.33, 0.5, 1, 2, 3), scales=(8,)):

    anchor_num = len(ratios) * len(scales)
    base_anchors = np.zeros((anchor_num, 4), dtype=np.float32)
    size = total_stride ** 2
    
    # Compute width and height for each anchor box aspect ratio
    idx = 0
    for r in ratios:
        w = int(np.sqrt(size / r))
        h = int(w * r)
        
        for s in scales:
            base_anchors[idx] = [0, 0, w * s, h * s]
            idx += 1
    
    # Copy the five anchor boxes across the 19×19 response map
    anchors = np.tile(base_anchors, response_sz * response_sz).reshape(-1, 4)

    # Pixel offsets relative to image center
    begin = -(response_sz // 2) * total_stride
    xs, ys = np.meshgrid(begin + total_stride * np.arange(response_sz), begin + total_stride * np.arange(response_sz))
    
    # Assign offsets to each anchor
    xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
    ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
    anchors[:, 0] = xs
    anchors[:, 1] = ys

    return anchors.astype(np.float32)