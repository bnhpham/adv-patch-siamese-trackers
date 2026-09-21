import glob
import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.siamban.siamban_tracker import build_siamban_tracker, cfg
from utils import paste_rotated_patch, load_patch, make_video, paste_patch_on_frame


# SiamBAN analogue of the other evaluate_video_*.py scripts' update_with_patch().
# Paste the patch into the cropped/resized search image before inference, then reproduces siamban's SiamBANTracker.track() update.
#
# This code is adapted from SiamBAN's SiamBANTracker.track() (Copyright (c) SenseTime, Apache License 2.0, see models/siamban/LICENSE.txt)
# and reuses the real SiamBAN tracker's own crop/decode helpers (get_subwindow, points, window, _convert_score/_convert_bbox, _bbox_clip).
# We modified the code such that the patch is pasted into the cropped search image before the forward pass.
def update_with_patch(tracker, image, patch, patch_x, patch_y, patch_rotation, patch_size):

    image = np.asarray(image)
    track_cfg = cfg.TRACK

    w_z = tracker.size[0] + track_cfg.CONTEXT_AMOUNT * np.sum(tracker.size)
    h_z = tracker.size[1] + track_cfg.CONTEXT_AMOUNT * np.sum(tracker.size)
    s_z = np.sqrt(w_z * h_z)
    scale_z = track_cfg.EXEMPLAR_SIZE / s_z
    s_x = s_z * (track_cfg.INSTANCE_SIZE / track_cfg.EXEMPLAR_SIZE)

    x_crop = tracker.get_subwindow(image, tracker.center_pos, track_cfg.INSTANCE_SIZE, round(s_x), tracker.channel_average)

    # Add patch into the search image
    if patch is not None:
        patch_for_tracker = patch.to(device=x_crop.device, dtype=x_crop.dtype)

        # Scale patch from [0, 1] to [0, 255]
        if x_crop.max() > 1.0 and patch_for_tracker.max() <= 1.0:
            patch_for_tracker = patch_for_tracker * 255.0

        # RGB -> BGR
        patch_for_tracker = patch_for_tracker[:, [2, 1, 0], :, :]

        patch_resized = F.interpolate(patch_for_tracker, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
        x_crop_patched = paste_rotated_patch(x_crop, patch_resized, patch_x, patch_y, patch_rotation)
    else:
        x_crop_patched = x_crop

    with torch.no_grad():
        tracker.model.eval()
        outputs = tracker.model.track(x_crop_patched)

    score = tracker._convert_score(outputs["cls"])
    pred_bbox = tracker._convert_bbox(outputs["loc"], tracker.points)

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) / sz(tracker.size[0] * scale_z, tracker.size[1] * scale_z))
    r_c = change((tracker.size[0] / tracker.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :]))

    penalty = np.exp(-(r_c * s_c - 1) * track_cfg.PENALTY_K)
    pscore = penalty * score
    pscore = pscore * (1 - track_cfg.WINDOW_INFLUENCE) + tracker.window * track_cfg.WINDOW_INFLUENCE
    best_idx = np.argmax(pscore)

    bbox = pred_bbox[:, best_idx] / scale_z
    lr = penalty[best_idx] * score[best_idx] * track_cfg.LR

    cx = bbox[0] + tracker.center_pos[0]
    cy = bbox[1] + tracker.center_pos[1]
    width = tracker.size[0] * (1 - lr) + bbox[2] * lr
    height = tracker.size[1] * (1 - lr) + bbox[3] * lr

    cx, cy, width, height = tracker._bbox_clip(cx, cy, width, height, image.shape[:2])

    tracker.center_pos = np.array([cx, cy])
    tracker.size = np.array([width, height])

    box = np.array([cx - width / 2, cy - height / 2, width, height])

    return x_crop_patched, box, s_x


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/person-7/img")
    parser.add_argument("--weights", default="models/siamban/model.pth")
    parser.add_argument("--config", default="models/siamban/config.yaml")
    parser.add_argument("--patch", default=None)
    parser.add_argument("--out_dir", default="results/siamban/video/person-7")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=2819)
    parser.add_argument("--init_x", type=float, default=557)
    parser.add_argument("--init_y", type=float, default=330)
    parser.add_argument("--init_w", type=float, default=83)
    parser.add_argument("--init_h", type=float, default=202)
    parser.add_argument("--patch_x", type=int, default=80)
    parser.add_argument("--patch_y", type=int, default=170)
    parser.add_argument("--patch_rotation", type=float, default=0)
    parser.add_argument("--patch_size", type=int, default=60)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    search_images_dir = os.path.join(args.out_dir, "search_images")
    os.makedirs(search_images_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Initialize tracker
    tracker = build_siamban_tracker(net_path=args.weights, config_path=args.config)
    first_frame = cv2.imread(os.path.join(args.frames, f"{args.start:08d}.jpg"))
    tracker.init(first_frame, np.array([args.init_x, args.init_y, args.init_w, args.init_h]))

    patch = load_patch(args.patch, tracker.device) if args.patch is not None else None

    patch_x, patch_y, patch_rotation, patch_size = args.patch_x, args.patch_y, args.patch_rotation, args.patch_size

    for i in range(args.start, args.end + 1):

        # One can set the trajectory of the patch by changing the its location, rotation and size at specific frames
        
        # trajectory adv patch
        if i == 50:
            patch_x += 35
            patch_y += 35
        if i == 100:
            patch_x += 30
            patch_y += 30
        if i == 410:
            patch_x += 45
            patch_y += 5
        if i == 530:
            patch_x -= 50
            patch_y += 30
        if i == 630:
            patch_x += 30
            patch_y -= 20

        # Capture the crop parameters BEFORE update_with_patch overwrites them, since these are the
        # exact center/size used to build this frame's search image crop.
        crop_center_xy = tracker.center_pos.copy()

        frame = cv2.imread(os.path.join(args.frames, f"{i:08d}.jpg"))
        patched_image, box, crop_s_x = update_with_patch(tracker, frame, patch, patch_x, patch_y, patch_rotation, patch_size)
        gx, gy, gw, gh = box.astype(int)

        # Paste the patch into the original frame and draw the bounding box on top
        out_sz = patched_image.shape[-1]
        if patch is not None:
            frame_patched = paste_patch_on_frame(frame, patch, crop_center_xy, crop_s_x, out_sz, patch_x, patch_y, patch_rotation, patch_size, tracker.device)
        else:
            frame_patched = frame.copy()
        cv2.rectangle(frame_patched, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(frames_dir, f"{i:08d}_frame.jpg"), frame_patched)

        # Convert patched search tensor from BCHW to HWC for visualization
        img_np = patched_image[0].detach().cpu().permute(1, 2, 0).numpy()

        # Convert global (g) coordinates in the original frame into local (l) coordinates inside the cropped search image
        gx, gy, gw, gh = box
        center_x, center_y = tracker.center_pos[0], tracker.center_pos[1]

        lx = int((gx - (center_x - crop_s_x / 2)) * out_sz / crop_s_x)
        ly = int((gy - (center_y - crop_s_x / 2)) * out_sz / crop_s_x)
        lw = int(gw * out_sz / crop_s_x)
        lh = int(gh * out_sz / crop_s_x)

        cv2.rectangle(img_np, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(search_images_dir, f"{i:08d}.jpg"), img_np)

        if i % 50 == 0:
            print(i)

    # Create videos
    make_video(search_images_dir, os.path.join(search_images_dir, "tracking.mp4"), fps=30)
    make_video(frames_dir, os.path.join(frames_dir, "tracking_frame.mp4"), fps=30)


if __name__ == "__main__":
    main()

    # Transferability test against SiamBAN (siamban_r50_l234):
    # python .\evaluate_video_siamban.py --init_x 557 --init_y 330 --init_w 83 --init_h 202 --patch patch/patch_best.pt --start 1 --end 1000 --patch_x 20 --patch_y 20
