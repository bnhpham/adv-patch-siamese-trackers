"""
Adversarial patch construction for hijacking a SiamRPN tracker
inspired by the work of Nakka et al., "Universal, Transferable Adversarial Perturbations for Visual Object Trackers" (ECCV 2022).

Paper: https://link.springer.com/chapter/10.1007/978-3-031-25056-9_27
Github: https://github.com/krishnakanthnakka/TTAttack

The authors crafted an universal image perturbation that forces the tracker to follow a predefined trajectory across all frames 
of an input video. Although they optimize an imperceptible perturbation over the entire image rather than a small printable patch,
we adopt their loss function as it directly optimizes the tracker towards a desired target location, which is the patch position.
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.siamrpn.siamrpn import SiamRPN
from utils import paste_rotated_patch, foreground_scores, build_anchors, save_tensor_as_image, gaussian_blur_patch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")
SEED=1234
ANCHORS_NP = build_anchors()
ANCHORS_T = torch.from_numpy(ANCHORS_NP).to(device=device)


# Convert search-image pixel coordinates to response-map coordinates
def pixel_to_cell(px: float, py: float):

    cell_x = int(round((px - 135.5) / 8.0 + 9))
    cell_y = int(round((py - 135.5) / 8.0 + 9))
    return cell_x, cell_y


# Create fixed evaluation set so every epoch is compared under exactly the same samples, positions and angles.
def build_eval_set(dataset, n_samples=50, patch_size=60, max_offset_px=64.0, seed=1234, angles=[-90.0, 90.0]):

    # Pick random indices
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(dataset), size=n_samples, replace=False)

    eval_set = []

    for sample_idx in sample_indices:
        for angle_deg in angles:
            while True:
                x0 = int(rng.integers(0, 271 - patch_size + 1))
                y0 = int(rng.integers(0, 271 - patch_size + 1))

                patch_cx = x0 + patch_size / 2.0
                patch_cy = y0 + patch_size / 2.0

                patch_x, patch_y = pixel_to_cell(patch_cx, patch_cy)

                #desired_x, desired_y = desired_cell_from_angle(patch_cx, patch_cy, angle_deg, max_offset_px)
                desired_x, desired_y = patch_x, patch_y 

                valid = (0 <= patch_x <= 18 and 0 <= patch_y <= 18 and 0 <= desired_x <= 18 and 0 <= desired_y <= 18)

                if valid:
                    break

            eval_set.append({"sample_idx": int(sample_idx), "angle_deg": float(angle_deg), "x0": x0, "y0": y0, 
                             "patch_x": patch_x, "patch_y": patch_y, "desired_x": desired_x, "desired_y": desired_y})

    return eval_set


# Evaluate how closely the predicted cells follow the desired cells on the evaluation set
# The lower mean_distance, the better. The higher exact_rate and within_one_rate, the better.
@torch.no_grad()
def evaluate_patch(tracker, dataset, patch, eval_set, device, debug=False):

    tracker.eval()

    distances = []
    exact_hits = 0
    within_one_hits = 0

    for i, eval_sample in enumerate(eval_set):
        template, x = dataset[eval_sample["sample_idx"]]

        template = template.unsqueeze(0).to(device)
        x = x.unsqueeze(0).to(device)

        x_adv = paste_rotated_patch(search=x, patch=patch, x_pos=eval_sample["x0"], y_pos=eval_sample["y0"], angle_deg=eval_sample["angle_deg"])

        out_cls, out_reg = tracker(template, x_adv)
        fg = foreground_scores(out_cls)  # (1, 5, 19, 19)

        _, num_anchors, map_h, map_w = fg.shape
        cells_per_anchor = map_h * map_w

        # Predicted cell
        best_flat = int(torch.argmax(fg[0].reshape(-1)).item())
        pred_anchor = best_flat // cells_per_anchor
        cell_flat = best_flat % cells_per_anchor

        pred_y = cell_flat // map_w
        pred_x = cell_flat % map_w

        # Target cell
        desired_x = eval_sample["desired_x"]
        desired_y = eval_sample["desired_y"]

        # Compute distance between target and predicted
        dist_x = pred_x - desired_x
        dist_y = pred_y - desired_y
        distance = float(np.sqrt(dist_x * dist_x + dist_y * dist_y))
        distances.append(distance)

        if pred_x == desired_x and pred_y == desired_y:
            exact_hits += 1

        if abs(dist_x) <= 1 and abs(dist_y) <= 1:
            within_one_hits += 1

        if debug and i < 10:
            print(
                f"[BEST_EVAL case={i:03d}] "
                f"sample={eval_sample['sample_idx']} "
                f"angle={eval_sample['angle_deg']:+.0f} "
                f"patch=(x={eval_sample['patch_x']},y={eval_sample['patch_y']}) "
                f"desired=(x={desired_x},y={desired_y}) "
                f"pred=(x={pred_x},y={pred_y}) "
                f"anchor={pred_anchor} "
                f"distance={distance:.2f}"
            )

    distances = np.asarray(distances, dtype=np.float32)
    total = len(eval_set)

    return {
        "mean_distance": float(distances.mean()),
        "median_distance": float(np.median(distances)),
        "max_distance": float(distances.max()),
        "exact_rate": exact_hits / total,
        "within_one_rate": within_one_hits / total,
        "num_cases": total,
    }


# Dataloader
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.templates = data["templates"]
        self.searches = data["searches"]

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, idx):
        z = torch.from_numpy(self.templates[idx]).float()
        x = torch.from_numpy(self.searches[idx]).float()

        if z.shape[-1] == 3:
            z = z.permute(2, 0, 1)
        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1)

        if z.max() > 1.5:
            z = z / 255.0
        if x.max() > 1.5:
            x = x / 255.0

        return z, x
    
# Tracker wrapper
class SiamRPNWrapper(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.model = SiamRPN(anchor_num=5)
        state = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, template, search):

        # Process a batch of template–search image pairs simultaneously instead of processing one pair at a time
        B = template.shape[0]
        kernel_reg, kernel_cls = self.model.learn(template)
        out_reg, out_cls = self.model.inference(search, kernel_reg, kernel_cls, B)
        return out_cls, out_reg


def train(args):
    
    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataLoader & SiamRPN tracker
    dataset = TrainDataset(args.data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    tracker = SiamRPNWrapper(args.weights).to(device)

    # Patch & Optimizer
    raw_patch = torch.zeros(1, 3, 60, 60, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw_patch], lr=args.lr)

    step = 0
    eval_angles = [-90.0, 0, 90.0]

    # Build evaluation set to track the progress
    eval_set = build_eval_set(dataset=dataset, n_samples=25, patch_size=60, max_offset_px=64.0, seed=SEED, angles=eval_angles)
    best_mean_distance = float("inf")
    best_exact_rate = -1.0
    best_epoch = -1
    print(f"checkpoint evaluation set={len(eval_set)}")

    for epoch in range(args.epochs):
        for z, x in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            z = z.to(device) # template
            x = x.to(device) # search image

            # Map raw_patch to valid RGB pixel values in [0,1]
            patch = torch.sigmoid(raw_patch)

            # Sample random patch size (60px, 45px or 30px) and resize patch (to make the patch more robust to distance)
            scale = random.choice([0.5, 0.75, 1.0])
            patch_size = int(60 * scale)
            patch_resized = F.interpolate(patch, size=(patch_size, patch_size), mode="bilinear", align_corners=False)

            # 50% of training includes blurring
            # One might also add brightness and contrast augmentation to make the patch more robust to augmentation
            if random.random() >= 0.5:
                patch_resized = gaussian_blur_patch(patch=patch_resized, ksize=3)

            H, W = x.shape[-2], x.shape[-1]

            # Sampling rotation angle
            angle_deg = random.randint(-90, 90)

            # Reject training samples where the patch position would leave the response map --> resample another position 
            while True:
                x1 = torch.randint(0, W - patch_size + 1, (1,), device=device).item()
                y1 = torch.randint(0, H - patch_size + 1, (1,), device=device).item()

                # Patch center in image coordinates
                patch_cx = x1 + patch_size / 2
                patch_cy = y1 + patch_size / 2

                # Map the patch center in pixel space to the nearest response map cell
                # "- 135.5" = Patch position relative to the center of the 271×271 search image
                # "/ 8" = Convert the pixel offset into response map cells as the tracker uses a stride of 8 pixels
                # "+ 9" = Shift the origin from the center (0,0) to the center cell of the 19×19 response map (9,9)
                # "round()" = Select the nearest response map cell
                patch_row = int(round((patch_cy - 135.5) / 8 + 9))
                patch_col = int(round((patch_cx - 135.5) / 8 + 9))

                if 0 <= patch_row <= 18 and 0 <= patch_col <= 18:
                    break
            
            # Add patch to search image
            x_adv = paste_rotated_patch(search=x, patch=patch_resized, x_pos=x1, y_pos=y1, angle_deg=angle_deg)

            # Define Tracker's target position.
            # In this script, the target should be sampled patch position.
            # However, one can also define a different position to enable a targeted attack (let the tracker focus a specific image region instead of the patch)
            desired_col = patch_col
            desired_row = patch_row

            # Clean tracker prediction before adding patch
            with torch.no_grad():
                clean_out_cls, clean_out_reg = tracker(z, x)
                clean_fg = foreground_scores(clean_out_cls)  # (B, 5, 19, 19)

            # Tracker prediction of image with adversarial patch
            out_cls, out_reg = tracker(z, x_adv)
            fg = foreground_scores(out_cls) # foreground probability for each of the 5 anchors at every response map cell
            
            #============================================================================
            # Shift loss (L_shift)
            #============================================================================

            # 1. Classifation loss
            desired_anchor = 2  # desired anchor (aspect ratio = 1 --> square)

            # Reshaphing
            B = x.shape[0]
            Hmap = Wmap = 19
            out_cls2 = out_cls.view(B, 2, 5, Hmap, Wmap)

            # Logits for target anchor (prediction whether target anchor is background or foreground)
            logits_t = out_cls2[0, :, desired_anchor, desired_row, desired_col].unsqueeze(0)  # shape: (1,2)
            
            # Define foreground as the desired class
            target_label = torch.tensor([1], dtype=torch.long, device=device)  # foreground

            # Cross-Entropy (negative log-likelihood of predicting the target)
            loss_shift_cls = F.cross_entropy(logits_t, target_label)

            # 2. Regression loss
            reg_pred_all = out_reg.view(B, 4, 5, Hmap, Wmap)

            # Extract the 4 predicted regression values for target anchor
            pred_reg = reg_pred_all[0, :, desired_anchor, desired_row, desired_col]  # shape: (4,)

            # Find target anchor
            flat_idx = desired_anchor * (Hmap * Wmap) + desired_row * Wmap + desired_col
            anchor = ANCHORS_T[flat_idx]
            anchor_cx, anchor_cy, anchor_w, anchor_h = anchor

            # Compute pixel offset of desired bounding-box center relative to the search image offset
            desired_cx = (desired_col - 9) * 8.0
            desired_cy = (desired_row - 9) * 8.0
            
            # Predicted bounding box should have the same size as the sampled patch size
            target_w = torch.tensor(float(patch_size), device=device)
            target_h = torch.tensor(float(patch_size), device=device)


            target_reg = torch.stack([
                (torch.tensor(desired_cx, device=device) - anchor_cx) / anchor_w,       # horizontal offset
                (torch.tensor(desired_cy, device=device) - anchor_cy) / anchor_h,       # vertical offset
                torch.log(target_w / anchor_w),                                         # scale adjustment for width
                torch.log(target_h / anchor_h),                                         # scale adjustment for height
            ])

            # L1 loss between predicted and desired regression vector
            loss_shift_reg = F.smooth_l1_loss(pred_reg, target_reg)


            #============================================================================
            # Standard loss (L_fool)
            #============================================================================
            
            tau = 0.20
            mu_c = -0.5
            mu_w = -1.0
            mu_h = -1.0
            lambda_fool_cls = 10.0
            lambda_fool_reg = 1.0

            B, A, Hmap, Wmap = fg.shape

            # Examine foreground probabilities from the clean unpatched image 
            # and consider only anchors with a foreground probability more than threshold tau
            clean_target_mask = clean_fg > tau

            # Do not suppress the anchor that L_shift explicitly activates.
            clean_target_mask[0, desired_anchor, desired_row, desired_col] = False

            if clean_target_mask.any():
                # First term of formula (1): classification fool term
                cls_adv = out_cls.view(B, 2, A, Hmap, Wmap)
                prob_adv = F.softmax(cls_adv, dim=1)

                # Background and foreground probability of each anchor
                bg_adv = prob_adv[:, 0]
                fg_adv = prob_adv[:, 1]

                # Corresponds to H_i(j) - (1 - H_i(j))
                cls_gap = fg_adv - bg_adv

                # Minimizing loss --> fg_adv decreases and bg_adv increases
                # Once the gap is below mu_c, foreground probability will not be pushed down to 0 anymore
                loss_fool_cls = torch.clamp_min(cls_gap[clean_target_mask], mu_c).mean()

                # Second term of formula (1): Regression fool term
                reg_adv = out_reg.view(B, 4, A, Hmap, Wmap)

                # width and height regression values
                dw_adv = reg_adv[:, 2]
                dh_adv = reg_adv[:, 3]

                # If width and height regression values decrease, then width and height of bounding box decreases as well
                # bounding box shrinks until regression values reach -1
                loss_fool_reg = (torch.clamp_min(dw_adv[clean_target_mask], mu_w) + torch.clamp_min(dh_adv[clean_target_mask], mu_h)).mean()

            else:
                loss_fool_cls = torch.zeros((), device=device)
                loss_fool_reg = torch.zeros((), device=device)

            loss_fool = lambda_fool_cls * loss_fool_cls + lambda_fool_reg * loss_fool_reg

            #============================================================================
            # Total loss
            #============================================================================

            lambda_shift_cls = 200.0
            lambda_shift_reg = 100.0
            lambda_fool = 200.0
            loss = lambda_fool * loss_fool + lambda_shift_cls * loss_shift_cls + lambda_shift_reg * loss_shift_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.debug_every == 0:
                with torch.no_grad():
                    fg_map = fg[0].max(dim=0).values
                    pred_idx = int(torch.argmax(fg_map).item())

                    patch_x_cell = int(round((patch_cx - 135.5) / 8 + 9))
                    patch_y_cell = int(round((patch_cy - 135.5) / 8 + 9))
                    patch_x_cell = int(np.clip(patch_x_cell, 0, 18))
                    patch_y_cell = int(np.clip(patch_y_cell, 0, 18))

                    pred_y = pred_idx // 19
                    pred_x = pred_idx % 19

                    print(
                        f"step={step:05d} epoch={epoch+1} angle={angle_deg} "
                        f"patch_cell=(x={patch_x_cell}, y={patch_y_cell}) "
                        f"desired_cell=(x={desired_col}, y={desired_row}) "
                        f"pred_cell=(x={pred_x}, y={pred_y}) "
                        f"loss={loss.item():.4f} "
                        f"patch[min,max,std]=[{patch.min().item():.3f},{patch.max().item():.3f},{patch.std().item():.3f}]"
                    )

            step += 1

        # Save epoch checkpoint
        torch.save(
            {
                "raw_patch": raw_patch.detach().cpu(),
                "patch": torch.sigmoid(raw_patch).detach().cpu(),
                "epoch": epoch + 1,
            },
            out_dir / f"patch_epoch{epoch+1:03d}.pt"
        )

        # Save best patch so far
        patch_eval = torch.sigmoid(raw_patch).detach()
        metrics = evaluate_patch(tracker=tracker, dataset=dataset, patch=patch_eval, eval_set=eval_set, device=device, debug=(epoch + 1) % 10 == 0)
        print(
            f"[CHECKPOINT_EVAL] epoch={epoch + 1:03d} "
            f"mean_dist={metrics['mean_distance']:.3f} "
            f"median_dist={metrics['median_distance']:.3f} "
            f"exact={metrics['exact_rate']:.3f} "
            f"within1={metrics['within_one_rate']:.3f} "
            f"max_dist={metrics['max_distance']:.3f}"
        )
        # Primary criterion: minimum mean cell distance.
        # Tie-breaker: maximum exact-hit rate.
        is_better = (metrics["mean_distance"] < best_mean_distance - 1e-6 
                     or (abs(metrics["mean_distance"] - best_mean_distance) <= 1e-6 and metrics["exact_rate"] > best_exact_rate))
        
        if is_better:
            best_mean_distance = metrics["mean_distance"]
            best_exact_rate = metrics["exact_rate"]
            best_epoch = epoch + 1
            best_patch = patch_eval.cpu()
            torch.save(
                {
                    "patch": best_patch,
                    "raw_patch": raw_patch.detach().cpu(),
                    "epoch": best_epoch,
                    "metrics": metrics,
                    "eval_set": eval_set,
                },
                out_dir / "patch_best.pt",
            )
            save_tensor_as_image(best_patch, out_dir / "patch_best.png")
            print(f"[NEW BEST] epoch={best_epoch} mean_dist={best_mean_distance:.3f} exact={best_exact_rate:.3f} -> {out_dir / 'patch_best.pt'}")

    # Save patch at final epoch
    torch.save({"patch": torch.sigmoid(raw_patch).detach().cpu()}, out_dir / "patch_final.pt")
    print(f"saved final delta to {out_dir / 'patch_final.pt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/train.npz")
    parser.add_argument("--weights", default="models/siamrpn/model.pth")
    parser.add_argument("--out_dir", default="patch")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--debug_every", type=int, default=20)

    args = parser.parse_args()
    train(args)

    # To run the code:
    # python train_ttattack_phantom_adapter.py --epochs 100 --lr 0.005 --debug_every 100                     