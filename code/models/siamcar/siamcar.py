"""
Thin loader around the official SiamCAR tracker implementation
proposed by Guo et al., "SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking" (CVPR 2020). 

Paper: https://arxiv.org/abs/2003.06761
Code: https://github.com/ohhhyeahhh/SiamCAR

The vendored "pysot/" package under this directory uses absolute imports like "from pysot.core.config import cfg",
so this directory is added to sys.path before importing it, matching how the upstream repo expects PYTHONPATH to be set.
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker

DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config.yaml")
DEFAULT_WEIGHTS = os.path.join(_THIS_DIR, "model_general.pth")


# Loads the raw SiamCAR ModelBuilder (backbone + neck + car_head),
# for callers that want direct access to those submodules instead of the stateful, single-instance SiamCARTracker.
def build_siamcar_model(net_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG, device=None):

    cfg.merge_from_file(config_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelBuilder()

    checkpoint = torch.load(net_path, map_location="cpu", weights_only=False)

    # This checkpoint may be a plain state_dict, or a full training checkpoint (state_dict/epoch/optimizer).
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Older checkpoints trained with nn.DataParallel prefix every key with "module.".
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    # Some checkpoints (e.g. the LaSOT snapshot) were saved from an older code revision that named the
    # head submodule "rpn_head" instead of "car_head" - same structure, just renamed upstream at some point.
    state_dict = {("car_head." + k[len("rpn_head."):] if k.startswith("rpn_head.") else k): v for k, v in state_dict.items()}

    # Some checkpoints (e.g. the LaSOT snapshot) were saved from an older code revision where certain
    # 1x1 "channel-mixing" layers (e.g. ModelBuilder.down) used nn.Conv2d instead of nn.ConvTranspose2d -
    # mathematically the same operation for a 1x1/stride-1 layer, but with the weight's first two
    # dimensions swapped. load_state_dict(strict=False) still hard-crashes on a shape mismatch (strict
    # only covers missing/unexpected keys), so fix up any such transposable mismatch here first, and
    # drop anything else unresolvable so it surfaces as a "missing" key instead of crashing the load.
    model_state = model.state_dict()
    fixed_state_dict = {}
    for key, tensor in state_dict.items():
        expected_shape = model_state[key].shape if key in model_state else None
        if expected_shape is None or tensor.shape == expected_shape:
            fixed_state_dict[key] = tensor
        elif tensor.dim() >= 2 and tensor.transpose(0, 1).shape == expected_shape:
            print(f"[siamcar] transposing '{key}' from {tuple(tensor.shape)} to {tuple(expected_shape)} "
                  f"(Conv2d/ConvTranspose2d weight-layout mismatch)")
            fixed_state_dict[key] = tensor.transpose(0, 1).contiguous()
        else:
            print(f"[siamcar] dropping '{key}': checkpoint shape {tuple(tensor.shape)} incompatible "
                  f"with model shape {tuple(expected_shape)}")
    state_dict = fixed_state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # A severe mismatch (most of the model's parameters unmatched) means config_path and net_path almost certainly belong together.
    # Fail loudly instead of silently tracking with a mostly-randomly-initialized network.
    total_keys = len(model.state_dict())
    if len(missing) > total_keys // 2:
        raise RuntimeError(f"[siamcar] refusing to continue: only {total_keys - len(missing)}/{total_keys} "
                            f"parameters matched between '{config_path}' and '{net_path}' - these almost "
                            f"certainly don't belong together (check --variant/--config/--weights pairing).")

    if missing or unexpected:
        print(f"[siamcar] state_dict mismatch - missing={missing}, unexpected={unexpected}")

    model.eval().to(device)
    return model


# Builds a real SiamCAR SiamCAR tracker (siamcar_r50 by default), ready for .init(img, bbox).
def build_siamcar_tracker(net_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG, device=None):

    model = build_siamcar_model(net_path=net_path, config_path=config_path, device=device)
    tracker = SiamCARTracker(model, cfg.TRACK)
    tracker.hp = {"lr": cfg.TRACK.LR, "penalty_k": cfg.TRACK.PENALTY_K, "window_lr": cfg.TRACK.WINDOW_INFLUENCE}
    tracker.device = next(model.parameters()).device
    return tracker
