"""
Network architecture and tracking logic of the Distractor-aware Siamese Network (DaSiamRPN)
proposed by Wang et al., "Distractor-aware Siamese Networks for Visual Object Tracking" (ECCV 2018).

Paper: https://arxiv.org/abs/1808.06048
Code: https://github.com/foolwood/DaSiamRPN

Portions of this file are adapted from the official DaSiamRPN code, Copyright (c) 2018 Qiang Wang,
released under the MIT License (see LICENSE.txt in this folder).

Code was modified to run under current PyTorch without torch.autograd.Variable / hardcoded .cuda().
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from "code/net.py"
class SiamRPNBIG(nn.Module):

    def __init__(self, feature_out=512, anchor=5):
        super().__init__()

        configs = [3, 192, 512, 768, 768, 512]  # size=2 channel widths, matching SiamRPNBIG.model
        feat_in = configs[-1]

        self.featureExtract = nn.Sequential(nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
                                             nn.BatchNorm2d(configs[1]),
                                             nn.MaxPool2d(kernel_size=3, stride=2),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(configs[1], configs[2], kernel_size=5),
                                             nn.BatchNorm2d(configs[2]),
                                             nn.MaxPool2d(kernel_size=3, stride=2),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(configs[2], configs[3], kernel_size=3),
                                             nn.BatchNorm2d(configs[3]),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(configs[3], configs[4], kernel_size=3),
                                             nn.BatchNorm2d(configs[4]),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(configs[4], configs[5], kernel_size=3),
                                             nn.BatchNorm2d(configs[5]),)

        self.anchor = anchor
        self.feature_out = feature_out

        self.conv_r1 = nn.Conv2d(feat_in, feature_out * 4 * anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out * 2 * anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4 * anchor, 4 * anchor, 1)

        self.r1_kernel = None
        self.cls1_kernel = None

    # Cross-correlates the search-region features against the template kernels stored by temple().
    def forward(self, x):
        x_f = self.featureExtract(x)
        out_reg = self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel))
        out_cls = F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)
        return out_reg, out_cls

    # Builds the per-anchor correlation kernels from the exemplar (template) image and stores them.
    def temple(self, z):
        z_f = self.featureExtract(z)
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.shape[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor * 4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)


# Builds anchors for a score_size x score_size response map, tiled across `len(ratios)*len(scales)` anchors per cell.
# Columns are [x_offset, y_offset, w, h], offsets given relative to the search-crop center.
# Adapted from "code/run_SiamRPN.py".
def generate_anchor(total_stride, scales, ratios, score_size):

    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = total_stride * total_stride

    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = ws * scale
            anchor[count, 3] = hs * scale
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))

    ori = -(score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                          [ori + total_stride * dy for dy in range(score_size)])
    xx = np.tile(xx.flatten(), (anchor_num, 1)).flatten()
    yy = np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0] = xx.astype(np.float32)
    anchor[:, 1] = yy.astype(np.float32)

    return anchor


# Crops a `original_sz`x`original_sz` window centered at `pos` (0-indexed [x, y]) out of `im`,
# padding with `avg_chans` where the crop exceeds the image bounds, then resizes it to `model_sz`x`model_sz`.
# Returns a plain HWC numpy array (raw pixel scale, no normalization) - the caller converts to a tensor.
# Adapted from "code/utils.py".
def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):

    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1

    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin += left_pad
    context_xmax += left_pad
    context_ymin += top_pad
    context_ymax += top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    return im_patch


class TrackerDaSiamRPN:

    def __init__(self, net_path=None, device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TrackerConfig adapted fron "code/run_SiamRPN.py".
        self.cfg = {"exemplar_size": 127,
                    "total_stride": 8,
                    "context_amount": 0.5,
                    "ratios": [0.33, 0.5, 1, 2, 3],
                    "scales": [8],
                    "anchor_num": 5,
                    "penalty_k": 0.055,
                    "window_influence": 0.42,
                    "lr": 0.295,
                    "adaptive": True,}

        self.net = SiamRPNBIG()

        if net_path is not None:
            try:
                state_dict = torch.load(net_path, map_location="cpu", weights_only=True)
            except Exception:
                state_dict = torch.load(net_path, map_location="cpu", weights_only=False)

            missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[TrackerDaSiamRPN] state_dict mismatch - missing={missing}, unexpected={unexpected}")

        self.net = self.net.to(self.device)

    # SiamRPN_init adapted fron "code/run_SiamRPN.py".
    def init(self, image, box):
        image = np.asarray(image)

        # box is 1-indexed [x, y, w, h], same convention as TrackerSiamRPN.init
        x, y, w, h = box
        self.target_pos = np.array([x - 1 + (w - 1) / 2, y - 1 + (h - 1) / 2], dtype=np.float32)  # [cx, cy]
        self.target_sz = np.array([w, h], dtype=np.float32)  # [w, h]

        self.im_h, self.im_w = image.shape[:2]
        self.avg_chans = np.mean(image, axis=(0, 1))

        if self.cfg["adaptive"]:
            if (self.target_sz[0] * self.target_sz[1]) / float(self.im_w * self.im_h) < 0.004:
                self.instance_size = 287
            else:
                self.instance_size = 271
        else:
            self.instance_size = 271

        self.score_size = int((self.instance_size - self.cfg["exemplar_size"]) / self.cfg["total_stride"] + 1)
        self.anchors = generate_anchor(self.cfg["total_stride"], self.cfg["scales"], self.cfg["ratios"], self.score_size)

        window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        self.window = np.tile(window.flatten(), self.cfg["anchor_num"])

        context_amount = self.cfg["context_amount"]
        wc_z = self.target_sz[0] + context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + context_amount * sum(self.target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        z_crop = get_subwindow_tracking(image, self.target_pos, self.cfg["exemplar_size"], s_z, self.avg_chans)
        z_tensor = torch.from_numpy(z_crop).to(self.device).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            self.net.eval()
            self.net.temple(z_tensor)
