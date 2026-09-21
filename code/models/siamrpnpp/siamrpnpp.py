"""
Thin loader around the vendored official PySOT package to build the real SiamRPN++ tracker
proposed by Li et al., "SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks" (CVPR 2019).

Paper: https://arxiv.org/abs/1812.11703
Code: https://github.com/STVIR/pysot

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
from pysot.tracker.tracker_builder import build_tracker

DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config_resnet50.yaml")
DEFAULT_WEIGHTS = os.path.join(_THIS_DIR, "model_resnet50.pth")


# Loads the raw pysot ModelBuilder (backbone + neck + rpn_head), for callers that want direct access to
# those submodules (e.g. batched training) instead of the stateful, single-instance SiamRPNTracker.
def build_siamrpnpp_model(net_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG, device=None):

    cfg.merge_from_file(config_path)
    cfg.CUDA = torch.cuda.is_available() if device is None else (device.type == "cuda")
    device = device or torch.device("cuda" if cfg.CUDA else "cpu")

    model = ModelBuilder()

    state_dict = torch.load(net_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # A severe mismatch (most of the model's parameters unmatched) means config_path and net_path
    # almost certainly belong to different backbones/variants - fail loudly instead of silently
    # tracking with a mostly-randomly-initialized network.
    total_keys = len(model.state_dict())
    if len(missing) > total_keys // 2:
        raise RuntimeError(f"[siamrpnpp] refusing to continue: only {total_keys - len(missing)}/{total_keys} "
                            f"parameters matched between '{config_path}' and '{net_path}' - these almost "
                            f"certainly don't belong to the same backbone (check --backbone/--config/--weights pairing).")

    if missing or unexpected:
        print(f"[siamrpnpp] state_dict mismatch - missing={missing}, unexpected={unexpected}")

    model.eval().to(device)
    return model


# Builds a real pysot SiamRPNTracker (siamrpn_r50_l234_dwxcorr by default), ready for .init(img, bbox).
def build_siamrpnpp_tracker(net_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG, device=None):

    model = build_siamrpnpp_model(net_path=net_path, config_path=config_path, device=device)
    tracker = build_tracker(model)
    tracker.device = next(model.parameters()).device
    return tracker
