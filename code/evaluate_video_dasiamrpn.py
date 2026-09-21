import glob
import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.dasiamrpn.dasiamrpn import TrackerDaSiamRPN, get_subwindow_tracking
from utils import paste_rotated_patch, load_patch, make_video, paste_patch_on_frame


# DaSiamRPN analogue of evaluate_video_siamrpn.py's update_with_patch():
# Paste the patch into the cropped/resized search image before inference, then reproduces SiamRPNBIG's tracker_eval() update.
#
# The tracker_eval() logic is adapted from the official DaSiamRPN code
# (foolwood/DaSiamRPN, Copyright (c) 2018 Qiang Wang, MIT License, see models/dasiamrpn/LICENSE.txt).
# 
# Modified here to splice the patch into the search image before the forward pass.
def update_with_patch(tracker, image, patch, patch_x, patch_y, patch_rotation, patch_size):

    image = np.asarray(image)
    cfg = tracker.cfg
    context_amount = cfg["context_amount"]

    # Search-region crop size (note: axes swapped vs init(), matching the official implementation)
    wc_z = tracker.target_sz[1] + context_amount * sum(tracker.target_sz)
    hc_z = tracker.target_sz[0] + context_amount * sum(tracker.target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = cfg["exemplar_size"] / s_z
    d_search = (tracker.instance_size - cfg["exemplar_size"]) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop = get_subwindow_tracking(image, tracker.target_pos, tracker.instance_size, round(s_x), tracker.avg_chans)
    x_tensor = torch.from_numpy(x_crop).to(tracker.device).permute(2, 0, 1).unsqueeze(0).float()

    # Add patch into the search image
    if patch is not None:
        patch_for_tracker = patch.to(device=tracker.device, dtype=x_tensor.dtype)

        # Scale patch from [0, 1] to [0, 255]
        if x_tensor.max() > 1.0 and patch_for_tracker.max() <= 1.0:
            patch_for_tracker = patch_for_tracker * 255.0

        # RGB -> BGR
        patch_for_tracker = patch_for_tracker[:, [2, 1, 0], :, :]

        patch_resized = F.interpolate(patch_for_tracker, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
        x_tensor_patched = paste_rotated_patch(x_tensor, patch_resized, patch_x, patch_y, patch_rotation)
    else:
        x_tensor_patched = x_tensor

    with torch.no_grad():
        tracker.net.eval()
        delta, score = tracker.net(x_tensor_patched)

    anchors = tracker.anchors
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0)[1].data.cpu().numpy()

    delta[0] = delta[0] * anchors[:, 2] + anchors[:, 0]
    delta[1] = delta[1] * anchors[:, 3] + anchors[:, 1]
    delta[2] = np.exp(delta[2]) * anchors[:, 2]
    delta[3] = np.exp(delta[3]) * anchors[:, 3]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        wh_pad = (w + h) * 0.5
        return np.sqrt((w + wh_pad) * (h + wh_pad))

    target_sz_crop = tracker.target_sz * scale_z
    s_c = change(sz(delta[2], delta[3]) / sz(target_sz_crop[0], target_sz_crop[1]))
    r_c = change((target_sz_crop[0] / target_sz_crop[1]) / (delta[2] / delta[3]))

    penalty = np.exp(-(r_c * s_c - 1.) * cfg["penalty_k"])
    pscore = penalty * score
    pscore = pscore * (1 - cfg["window_influence"]) + tracker.window * cfg["window_influence"]
    best_id = np.argmax(pscore)

    target = delta[:, best_id] / scale_z
    lr = penalty[best_id] * score[best_id] * cfg["lr"]

    res_x = target[0] + tracker.target_pos[0]
    res_y = target[1] + tracker.target_pos[1]
    res_w = tracker.target_sz[0] * (1 - lr) + target[2] * lr
    res_h = tracker.target_sz[1] * (1 - lr) + target[3] * lr

    res_x = np.clip(res_x, 0, tracker.im_w)
    res_y = np.clip(res_y, 0, tracker.im_h)
    res_w = np.clip(res_w, 10, tracker.im_w)
    res_h = np.clip(res_h, 10, tracker.im_h)

    tracker.target_pos = np.array([res_x, res_y])
    tracker.target_sz = np.array([res_w, res_h])

    box = np.array([tracker.target_pos[0] - tracker.target_sz[0] / 2,
                     tracker.target_pos[1] - tracker.target_sz[1] / 2,
                     tracker.target_sz[0],
                     tracker.target_sz[1]])

    return x_tensor_patched, box, s_x


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/person-7/img")
    parser.add_argument("--weights", default="models/dasiamrpn/SiamRPNBIG.model")
    parser.add_argument("--patch", default=None)
    parser.add_argument("--out_dir", default="results/dasiamrpn/video/person-7")
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
    tracker = TrackerDaSiamRPN(net_path=args.weights)
    first_frame = cv2.imread(os.path.join(args.frames, f"{args.start:08d}.jpg"))
    tracker.init(first_frame, np.array([args.init_x, args.init_y, args.init_w, args.init_h]))

    patch = load_patch(args.patch, tracker.device) if args.patch is not None else None

    patch_x, patch_y, patch_rotation, patch_size = args.patch_x, args.patch_y, args.patch_rotation, args.patch_size

    for i in range(args.start, args.end + 1):

        # One can set the trajectory of the patch by changing the its location, rotation and size at specific frames
        
        # Trajectory yellow patch
        """
        if i == 50:
            patch_x += 35
            patch_y += 35
        if i == 100:
            patch_x += 30
            patch_y += 30
        if i == 125:
            patch_x += 30
            patch_y += 30
        if i == 230:
            patch_x -= 20
            patch_y -= 20
        if i == 460:
            patch_y += 30
        if i == 540:
            patch_x += 40
            patch_y -= 5
        if i == 585:
            patch_x -= 20
            patch_y -= 10
        if i == 700:
            patch_x -= 15
            patch_y -= 10
        if i == 960:
            patch_x += 30
            patch_y += 30
        """

        # trajectory adv patch
        if i == 50:
            patch_x += 35
            patch_y += 35
        if i == 100:
            patch_x += 30
            patch_y += 30
        if i == 125:
            patch_x += 30
            patch_y += 30
        if i == 230:
            patch_x -= 20
            patch_y -= 20
        if i == 250:
            patch_x += 20
            patch_y += 15
        if i == 350:
            patch_x += 20
        if i == 500:
            patch_x -= 40
            patch_y += 5
        if i == 630:
            patch_y -= 5
        if i == 700:
            patch_y -= 10
        if i == 740:
            patch_y += 10
        if i == 860:
            patch_x += 20
        if i == 950:
            patch_y += 10
            patch_x += 20

        # Capture the crop parameters BEFORE update_with_patch overwrites them, since these are the
        # exact center/size used to build this frame's search image crop.
        crop_center_xy = tracker.target_pos.copy()

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
        center_x, center_y = tracker.target_pos[0], tracker.target_pos[1]

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

    # Transferability test against DaSiamRPN:
    # python .\evaluate_video_dasiamrpn.py --init_x 557 --init_y 330 --init_w 83 --init_h 202 --patch patch/patch_best.pt --start 1 --end 1000 --patch_x 20 --patch_y 20
