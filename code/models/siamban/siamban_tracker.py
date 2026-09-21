"""
Thin loader around the official SiamBAN tracker implementation
proposed by Chen et al., "Siamese Box Adaptive Network for Visual Tracking" (CVPR 2020). 


Paper: https://arxiv.org/abs/2003.06761
Code: https://github.com/hqucv/siamban 

The vendored "siamban/" package under this directory uses absolute imports like "from siamban.core.config import cfg",
so this directory is added to sys.path before importing it, matching how the upstream repo expects PYTHONPATH to be set.

Note: the wrapper module is deliberately NOT named "siamban.py" as that would collide with the vendored "siamban/" package
sitting right next to it on sys.path.
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker

DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config.yaml")
DEFAULT_WEIGHTS = os.path.join(_THIS_DIR, "model.pth")


# Loads the raw siamban ModelBuilder (backbone + neck + head),
# for callers that want direct access to those submodules instead of the stateful, single-instance SiamBANTracker.
def build_siamban_model(net_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG, device=None):

    cfg.merge_from_file(config_path)
    cfg.CUDA = torch.cuda.is_available() if device is None else (device.type == "cuda")
    device = device or torch.device("cuda" if cfg.CUDA else "cpu")

    model = ModelBuilder()

    checkpoint = torch.load(net_path, map_location="cpu", weights_only=False)

    # This checkpoint may be a plain state_dict, or a full training checkpoint (state_dict/epoch/optimizer).
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Older checkpoints trained with nn.DataParallel prefix every key with "module.".
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[siamban] state_dict mismatch - missing={missing}, unexpected={unexpected}")

    model.eval().to(device)
    return model


# Builds a real siamban SiamBANTracker (siamban_r50_l234 by default), ready for .init(img, bbox).
def build_siamban_tracker(net_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG, device=None):

    model = build_siamban_model(net_path=net_path, config_path=config_path, device=device)
    tracker = build_tracker(model)
    tracker.device = next(model.parameters()).device
    return tracker
