# Third-party notices

This project builds on published tracking research and code. The parts below are used under their original licenses.

| Project | License | What is used here | License text |
|---|---|---|---|
| [DaSiamRPN](https://github.com/foolwood/DaSiamRPN) (Copyright (c) 2018 Qiang Wang) | MIT | `code/models/dasiamrpn/dasiamrpn.py` is adapted from the official network and tracker code (modified to run on current PyTorch). `update_with_patch` in `code/evaluate_video_dasiamrpn.py` is adapted from its `tracker_eval()`. | `code/models/dasiamrpn/LICENSE.txt` |
| [PySOT / SiamRPN++](https://github.com/STVIR/pysot) (Copyright (c) SenseTime) | Apache 2.0 | `code/models/siamrpnpp/config_*.yaml` are unmodified copies of PySOT's experiment configs. `update_with_patch` in `code/evaluate_video_siamrpnpp.py` is adapted from `SiamRPNTracker.track()` and modified to paste the adversarial patch into the search image. The PySOT package itself is not included (see README). | `code/models/siamrpnpp/PYSOT_LICENSE` |
| [SiamBAN](https://github.com/hqucv/siamban) (Copyright (c) SenseTime) | Apache 2.0 | `code/models/siamban/config.yaml` is an unmodified copy of SiamBAN's experiment config. `update_with_patch` in `code/evaluate_video_siamban.py` is adapted from `SiamBANTracker.track()` and modified to paste the adversarial patch into the search image. The SiamBAN package itself is not included (see README). | `code/models/siamban/LICENSE.txt` |

## Code that is not redistributed

The following upstream repositories do not publish a license. Files derived from them are therefore not part of this repository, and users are pointed to the original sources instead (see "Files not included" in the README).

| Project | What is not included |
|---|---|
| [siamrpn-pytorch](https://github.com/huanglianghua/siamrpn-pytorch) (huanglianghua) | `code/models/siamrpn/siamrpn.py` and `code/evaluate_video_siamrpn.py`. `code/create_patch.py` and `code/evaluate_image_siamrpn.py` import the upstream `SiamRPN` class from `siamrpn.py`. |
| [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR) (ohhhyeahhh) | `code/evaluate_video_siamcar.py` and `code/models/siamcar/config.yaml`. The SiamCAR package itself is not included either. |

Pretrained weights are not distributed with this repository; each one has to be downloaded from its original source under that source's terms (see README).

## Ideas and papers

The adversarial-patch loss in `code/create_patch.py` follows the idea of Nakka et al., "Universal, Transferable Adversarial Perturbations for Visual Object Trackers" (ECCV 2022). The tracker architectures come from the papers cited in the README.
