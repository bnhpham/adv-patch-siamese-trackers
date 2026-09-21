import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.siamrpn.siamrpn import SiamRPN
from utils import paste_rotated_patch, load_patch, foreground_scores, build_anchors, save_tensor_as_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INSTANCE_SIZE = 271
EXEMPLAR_SIZE = 127
TOTAL_STRIDE = 8
RATIOS = (0.33, 0.5, 1.0, 2.0, 3.0)
SCALES = (8.0,)
PENALTY_K = 0.055
WINDOW_INFLUENCE = 0.42
TRACK_LR = 0.295
ANCHORS_NP = build_anchors()
ANCHORS_T = torch.from_numpy(ANCHORS_NP).to(device=DEVICE)


@dataclass
class Winner:
    flat_index: int
    anchor: int
    row: int
    col: int
    score: float
    box_cxcywh: Tuple[float, float, float, float]


#============================================================================
# Auxiliary functions
#============================================================================

# Convert a HWC image in [0, 255] to a CHW tensor in [0, 1]
def to_chw_float01(arr: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(arr).float()
    x = x.permute(2, 0, 1)
    x = x / 255.0
    return x.clamp(0.0, 1.0)


# Converts a bounding box from coordinates relative to the center of the image ((0,0) is at image center)
# into image corner coordinates [x1, y1, x2, y2] in the search image ((0,0) is at the upper left corner)
def box_to_image_corners(box_cxcywh, image_h, image_w):
    center_x, center_y, width, height = box_cxcywh
    center_x += (image_w - 1) / 2.0
    center_y += (image_h - 1) / 2.0

    x1 = int(np.floor(center_x - width / 2.0))
    y1 = int(np.floor(center_y - height / 2.0))
    x2 = int(np.ceil(center_x + width / 2.0))
    y2 = int(np.ceil(center_y + height / 2.0))

    return (int(np.clip(x1, 0, image_w - 1)), int(np.clip(y1, 0, image_h - 1)), int(np.clip(x2, 0, image_w - 1)), int(np.clip(y2, 0, image_h - 1)))


def print_winner(label, winner):
    cx, cy, width, height = winner.box_cxcywh
    print(
        f"  {label} cell=(x={winner.col}, y={winner.row}) "
        f"anchor={winner.anchor} score={winner.score:.6f} "
        f"box_centered=(cx={cx:.2f}, cy={cy:.2f}, w={width:.2f}, h={height:.2f})"
    )

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


#============================================================================
# Raw winner
#============================================================================

def compute_raw_winner(out_reg, out_cls, search_size=271):
    """
    Convert the raw network outputs (out_reg, out_cls) into bounding boxes.
    Return bounding box with the highest foreground probability before any penalties and windowing.

    Returns:
        bbox_xywh:  Raw winner in search-image coordinates: [x, y, width, height].
        best_score: Raw foreground probability.
        best_id:    Flat proposal index.
        scores:     Foreground score for every anchor.
        boxes:      All boxes expressed relative to the center of the search image ((0,0) is the image center), ordered as [cx, cy, width, height].
    """

    anchors = ANCHORS_T

    # Converts the tracker's predicted offsets into actual coordinates and sizes relative to the search image center
    offsets = out_reg.permute(1, 2, 3, 0).contiguous().view(4, -1)

    decoded_cx = offsets[0] * anchors[:, 2] + anchors[:, 0]                 # x-coordinate of the center point of the bounding box
    decoded_cy = offsets[1] * anchors[:, 3] + anchors[:, 1]                 # y-coordinate of the center point of the bounding box
    decoded_w = torch.exp(offsets[2].clamp(max=10.0)) * anchors[:, 2]
    decoded_h = torch.exp(offsets[3].clamp(max=10.0)) * anchors[:, 3]

    boxes = torch.stack([decoded_cx, decoded_cy, decoded_w, decoded_h], dim=1)

    # Compute foreground probabilities (scores)
    scores = F.softmax(out_cls.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0)[1]

    # Select bounding box with highest foreground score
    best_id = int(torch.argmax(scores).item())
    best_score = float(scores[best_id].item())
    cx, cy, width, height = boxes[best_id]

    # Convert to search-image coordinates
    image_center = (search_size - 1) / 2.0
    image_cx = image_center + cx
    image_cy = image_center + cy

    x = image_cx - width / 2.0
    y = image_cy - height / 2.0
    bbox_xywh = torch.stack([x, y, width, height])

    return bbox_xywh, best_score, best_id, scores, boxes


#============================================================================
# Final winner
#============================================================================

# Recreation of the tracker's _create_penalty() function
# Essentially, it discourages boxes whose size or aspect ratio changes too much compared to the previous tracked object
def create_scale_ratio_penalty(boxes, target_width, target_height, penalty_k=PENALTY_K):

    def _padded_size(width, height, context=0.5):
        c = context * (width + height)
        return torch.sqrt((width + c) * (height + c))

    def _larger_ratio(r):
        eps = torch.finfo(r.dtype).eps
        r = r.clamp_min(eps)
        return torch.maximum(r, 1.0 / r)

    dtype, device = boxes.dtype, boxes.device
    target_w = torch.tensor(float(target_width), dtype=dtype, device=device)
    target_h = torch.tensor(float(target_height), dtype=dtype, device=device)

    source_size = _padded_size(target_w, target_h)
    destination_size = _padded_size(boxes[:, 2], boxes[:, 3])
    change_size = _larger_ratio(destination_size / source_size)

    source_ratio = target_w / target_h
    destination_ratio = boxes[:, 2] / boxes[:, 3].clamp_min(torch.finfo(dtype).eps)
    change_ratio = _larger_ratio(destination_ratio / source_ratio)

    # Return a penalty between 0 and 1.
    # Boxes with a similar size and aspect ratio receive a value close to 1, while boxes that differ a lot receive a smaller value,
    # reducing their final score before the best box is chosen.
    return torch.exp(-(change_ratio * change_size - 1.0) * penalty_k)


# Recreation of the tracker's hann window, but using PyTorch instead of Numpy (cf. init() function in siamrpn.py)
def create_hann_window(response_sz, anchor_num, device, dtype):
    one_dim = torch.hann_window(response_sz, periodic=False, device=device, dtype=dtype)
    window_2d = torch.outer(one_dim, one_dim).reshape(-1)
    return window_2d.repeat(anchor_num)


def make_winner(winner_index, winner_score, decoded_boxes, target_width, target_height, response_h, response_w, track_lr=TRACK_LR):

    # bounding box of winner
    winner_cx, winner_cy, winnerl_w, winner_h = decoded_boxes[winner_index]

    # Original scale update (cf. update() function of SiamRPN)
    lr = winner_score * float(track_lr)
    previous_w = torch.as_tensor(target_width, device=decoded_boxes.device, dtype=decoded_boxes.dtype)
    previous_h = torch.as_tensor(target_height, device=decoded_boxes.device, dtype=decoded_boxes.dtype)
    updated_w = (1.0 - lr) * previous_w + lr * winnerl_w
    updated_h = (1.0 - lr) * previous_h + lr * winner_h

    # Compute (row, column) location on the response map
    cells_per_anchor = response_h * response_w
    anchor = winner_index // cells_per_anchor
    cell = winner_index % cells_per_anchor
    row = cell // response_w
    col = cell % response_w

    return Winner(flat_index=winner_index, anchor=anchor, row=row, col=col, score=float(winner_score.item()),
                  box_cxcywh=(float(winner_cx.item()), float(winner_cy.item()), float(updated_w.item()), float(updated_h.item()),))


# Save images comparing raw and final winners
def save_winner_image(search_chw, raw_winner, final_winner, path):

    path.parent.mkdir(parents=True, exist_ok=True)

    image = search_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    image = np.rint(image * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    raw_box = box_to_image_corners(raw_winner.box_cxcywh, height, width)
    final_box = box_to_image_corners(final_winner.box_cxcywh, height, width)

    # Raw winner: red. Final winner after penalties/windowing: green.
    cv2.rectangle(image, raw_box[:2], raw_box[2:], (0, 0, 255), 2)
    cv2.rectangle(image, final_box[:2], final_box[2:], (0, 255, 0), 2)
    cv2.putText(image, "raw", (raw_box[0], max(14, raw_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "final", (final_box[0], max(14, final_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write tracker result to {path}")


#============================================================================
# Main evaluation function
#============================================================================

# Return both the raw winner and the final winner after penalty/windowing
@torch.no_grad()
def run_tracker(model, template_float01, search_float01, target_width, target_height, out_path):
  
    # tracker output
    kernel_reg, kernel_cls = model.learn(template_float01)
    out_reg, out_cls = model.inference(search_float01, kernel_reg, kernel_cls)

    foreground_probs = foreground_scores(out_cls)
    _, anchor_num, response_h, response_w = foreground_probs.shape

    # Raw winner
    raw_bbox_xywh, raw_score, raw_index, raw_scores, boxes = compute_raw_winner(out_reg, out_cls, search_size=search_float01.shape[-1])

    cells_per_anchor = response_h * response_w
    raw_anchor = raw_index // cells_per_anchor
    raw_cell = raw_index % cells_per_anchor
    raw_row = raw_cell // response_w
    raw_col = raw_cell % response_w

    raw_box_centered = tuple(float(value) for value in boxes[raw_index].detach().cpu().tolist())
    raw_winner = Winner(flat_index=raw_index, anchor=raw_anchor, row=raw_row, col=raw_col, score=raw_score, box_cxcywh=raw_box_centered)

    # Panalty & windowing
    penalty = create_scale_ratio_penalty(boxes, target_width=target_width, target_height=target_height,)
    hann_window = create_hann_window(response_h, anchor_num, raw_scores.device, raw_scores.dtype)
    penalized_scores = raw_scores * penalty
    final_scores = ((1.0 - WINDOW_INFLUENCE) * penalized_scores + WINDOW_INFLUENCE * hann_window)

    # Final winner
    winner_index = int(torch.argmax(final_scores).item())
    winner_score = final_scores[winner_index]
    final_winner = make_winner( winner_index=winner_index, winner_score=winner_score, decoded_boxes=boxes, target_width=target_width,
                                               target_height=target_height, response_h=response_h, response_w=response_w)

    # Save images
    save_winner_image(search_float01[0], raw_winner, final_winner, out_path)
    print("raw bbox [x,y,w,h]:", raw_bbox_xywh.detach().cpu().tolist())
    return raw_winner, final_winner


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.npz")
    parser.add_argument("--weights", default="models/siamrpn/model.pth")
    parser.add_argument("--patch", default="patches//patch_best.pt")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out_dir", default="results/siamprn/images")
    parser.add_argument("--patch_x", type=int, default=106)
    parser.add_argument("--patch_y", type=int, default=106)
    parser.add_argument("--patch_size", type=int, default=60, choices=[30, 45, 60])
    parser.add_argument("--patch_rotation", type=float, default=0.0)
    parser.add_argument("--aug_brightness", type=float, default=0.0)
    parser.add_argument("--aug_contrast", type=float, default=1.0)
    parser.add_argument("--aug_blur", type=int, default=0)
    parser.add_argument("--aug_jpeg", type=int, default=0)
    parser.add_argument("--target_w", type=float, default=None)
    parser.add_argument("--target_h", type=float, default=None)
    args = parser.parse_args()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load template, search image and patch
    data = np.load(args.data)
    template = to_chw_float01(data["templates"][args.idx]).unsqueeze(0).to(DEVICE)
    search = to_chw_float01(data["searches"][args.idx]).unsqueeze(0).to(DEVICE)    
    patch = load_patch(args.patch, DEVICE)
    target_width, target_height = args.target_w, args.target_h

    # Adjust patch size
    if patch.shape[-2:] != (args.patch_size, args.patch_size):
        patch = F.interpolate(patch, size=(args.patch_size, args.patch_size), mode="bilinear", align_corners=False)

    # Augmentation techniques
    if args.aug_brightness != 0.0 or args.aug_contrast != 1.0:
        patch = adjust_brightness_contrast(patch, args.aug_brightness, args.aug_contrast)
    if args.aug_blur > 1:
        patch = gaussian_blur_patch(patch, args.aug_blur)

    # Paste patch into search image
    adversarial_search = paste_rotated_patch(search, patch, args.patch_x, args.patch_y, args.patch_rotation)

    # Load tracker
    model = SiamRPN(anchor_num=len(RATIOS) * len(SCALES)).to(DEVICE)
    state_dict = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Let tracker predict clean and patched image
    clean_raw, clean_final = run_tracker(model, template, search, target_width, target_height,
                                         out_dir / f"sample_{args.idx}_clean_tracker_result.png")
    adv_raw, adv_final = run_tracker(model, template, adversarial_search, target_width, target_height,
                                     out_dir / f"sample_{args.idx}_adv_tracker_result.png")

    print(f"device: {DEVICE}")
    print(f"target size: width={target_width:.2f}, height={target_height:.2f}")
    print("\nCLEAN SEARCH")
    print_winner("raw", clean_raw)
    print_winner("final", clean_final)
    print("\nADVERSARIAL SEARCH")
    print_winner("raw", adv_raw)
    print_winner("final", adv_final)


if __name__ == "__main__":
    main()

    # To reproduce the result for the first patch, run:
    # evaluate_image_siamrpn.py --patch patch/patch_best.pt --patch_x 80 --patch_y 170 --patch_size 60 --patch_rotation 45 --target_w 30 --target_h 90 --idx 11