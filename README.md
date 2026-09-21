# Adversarial Patch Attack Against Siamese Visual Trackers

This project crafts a printable adversarial patch that hijacks a Siamese visual object tracker's attention away from its real target, and evaluates how well that patch transfers to other trackers it was never optimized against.

## Key Features

- **Patch Training**  (`create_patch.py`): Optimizes a small (e.g. 60x60px) adversarial patch against a SiamRPN tracker using a loss function inspired by the work "[Universal, Transferable Adversarial Perturbations for Visual Object Trackers](https://link.springer.com/chapter/10.1007/978-3-031-25056-9_27)" by Nakka et al. (ECCV 2022). The loss consists of a shift-loss pulling the tracker's predicted box toward the patch's own position as well as a fool-loss that suppresses the tracker's confidence in the true target. The training includes changes to scaling, rotation and blur to increase physical robustness.
- **Transferability Evaluation**: Testing the SiamRPN-trained patch on more refined and robust trackers, each driven by its own official implementation. Our experiment includes:
  | Tracker | Paper | GitHub |
  |---|---|---|
  | SiamRPN (Baseline model to train the patch) | [Li et al., CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) | [Link](https://github.com/huanglianghua/siamrpn-pytorch) |
  | DaSiamRPN | [Zhu et al., ECCV 2018](https://arxiv.org/abs/1808.06048) | [Link](https://github.com/foolwood/DaSiamRPN) |
  | SiamRPN++ (ResNet50 / AlexNet / MobileNetV2 backbones) | [Li et al., CVPR 2019](https://arxiv.org/abs/1812.11703) | [Link](https://github.com/STVIR/pysot) |
  | SiamBAN | [Chen et al., CVPR 2020](https://arxiv.org/abs/2003.06761) | [Link](https://github.com/hqucv/siamban) |
  | SiamCAR (general / GOT-10k / LaSOT trained variants) | [Guo et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_SiamCAR_Siamese_Fully_Convolutional_Classification_and_Regression_for_Visual_Tracking_CVPR_2020_paper.html) | [Link](https://github.com/ohhhyeahhh/SiamCAR) |
- **Video-based Visualization**:  Scripts for each tracker that paste the patch into the tracker's search-crop input (the view of what the network actually sees) and composites it into the full original frame.
- **Physical (Real-World) Test** (`evaluate_physical_attack.py`): Running any of the nine tracker variants on a live webcam stream or a recorded video, so a printed patch can be tested in front of a real camera.

## Digital Adversarial Patch Attack Example

The following footage shows a transferability test of SiamRPN-trained patch pasted digitally to steal a SiamRPN++ tracker's attention. MobileNetV2 was used as backbone.

https://github.com/user-attachments/assets/3e0efa89-8aab-487d-a06f-950ff720aa37

## Transferability Evaluation

The patch was tested in the digital as well as in the physical world.

| Tracker / variant | Digital World | Phyiscal World |
|---|---|---|
| SiamRPN | :white_check_mark: | :white_check_mark: |
| DaSiamRPN | :white_check_mark: | :white_check_mark: |
| SiamRPN++ ResNet50 | :x: | :white_check_mark: |
| SiamRPN++ AlexNet | :white_check_mark: | :heavy_minus_sign: |
| SiamRPN++ MobileNetV2 | :white_check_mark: | :heavy_minus_sign: |
| SiamBAN | :x: | :x: |
| SiamCAR (general) | :white_check_mark: | :heavy_minus_sign: |
| SiamCAR (GOT-10k) | :x: | :x: |
| SiamCAR (LaSOT) | :white_check_mark: | :white_check_mark: |

:white_check_mark: denotes that the patch generally works on the tracker, whereas :x: means that the patch has no effect on the tracker. :heavy_minus_sign: means that attacking is possible, but significantly more difficult and only under certain conditions, e.g., slow movements of the patch.

## Project File Overview

```
code/
├── create_patch.py                 # Trains the adversarial patch against the SiamRPN tracker
├── utils.py                        # Shared helpers: patch pasting/rotation, video writing, anchor math
├── evaluate_image_siamrpn.py       # Single-image tracker evaluation with the patch applied
├── evaluate_video_siamrpn.py       # Video evaluation against the SiamRPN tracker (not included, see below)
├── evaluate_video_dasiamrpn.py     # Transferability test against DaSiamRPN
├── evaluate_video_siamrpnpp.py     # Transferability test against SiamRPN++ (--backbone resnet50|alexnet|mobilenet)
├── evaluate_video_siamban.py       # Transferability test against SiamBAN
├── evaluate_video_siamcar.py       # Transferability test against SiamCAR (--variant general|lasot|got10k) (not included, see below)
├── evaluate_physical_attack.py     # Real-world test: runs a selected tracker on a live webcam/video with a printed patch
├── models/
│   ├── siamrpn/siamrpn.py          # SiamRPN network + tracker from huanglianghua/siamrpn-pytorch (not included, see below)
│   ├── dasiamrpn/dasiamrpn.py      # DaSiamRPN network + tracker (ported from the official repo)
│   ├── siamrpnpp/siamrpnpp.py      # Loader for the official pysot SiamRPN++ implementation
│   ├── siamban/siamban_tracker.py  # Loader for the official SiamBAN implementation
│   └── siamcar/siamcar.py          # Loader for the official SiamCAR implementation
├── data/                           # Tracking sequences (not included - see Setup)
├── patch/                          # Trained patch checkpoints (not included - generated by create_patch.py)
└── results/                        # Generated evaluation videos/images (not included - generated by the evaluate_* scripts)
```

Note: `models/siamrpnpp/`, `models/siamcar/`, and `models/siamban/` each also expect a vendored copy of their upstream tracker's source code alongside the loader script (`pysot/` or `siamban/`). These are not included in this repository and must be cloned additionally (see Setup & Installation).

## Files not included

Four files are deliberately not part of this repository as the upstream projects they are derived from do not publish a license, so we are not allowed to redistribute them:

| File | Why it is missing | Where to get the original |
|---|---|---|
| `code/models/siamrpn/siamrpn.py` | SiamRPN network from [huanglianghua/siamrpn-pytorch](https://github.com/huanglianghua/siamrpn-pytorch), which has no license | `siamrpn.py` in that repo (copy it to `code/models/siamrpn/siamrpn.py`) |
| `code/evaluate_video_siamrpn.py` | Modified copy of that SiamRPN's `update()` with the patch-pasting step added | Not available |
| `code/evaluate_video_siamcar.py` | Modified copy of `SiamCARTracker.track()` from [ohhhyeahhh/SiamCAR](https://github.com/ohhhyeahhh/SiamCAR), which has no license | Not available |
| `code/models/siamcar/config.yaml` | Copy of SiamCAR's experiment config | `experiments/siamcar_r50/config.yaml` in the SiamCAR repo |

What this means in practice:

- **Patch training works.** `create_patch.py` only needs the upstream `siamrpn.py` (place it as described above).
- **DaSiamRPN, SiamRPN++ and SiamBAN evaluations work** with the vendored packages and weights from the setup section since their licenses (MIT / Apache 2.0) allow redistribution with attribution.
- **The SiamRPN and SiamCAR video evaluations cannot be reproduced from this repository alone.** Every `evaluate_video_*.py` script follows the same pattern: Crop the search region as the tracker does, paste the patch into the crop, run the tracker's forward pass and decode step. The DaSiamRPN, SiamRPN++ and SiamBAN scripts show this pattern.

## Setup & Installation

### 1. Environment

Python 3.10+ with:

```bash
pip install torch torchvision opencv-python numpy yacs tqdm
```

(Install a CUDA-enabled `torch` build if you have a GPU.)

### 2. Tracking data

`data/train.npz` contains a total of 250 randomly shuffled template/search pairs, extracted from various frames across 13 different videos. The data was provided by TU Berlin and is not included in this repository due to copyright reasons. The file holds three arrays, each with 250 entries, and is structured as the following:

| Array | Shape | Description |
|---|---|---|
| `templates` | `(250, 127, 127, 3)` | `uint8` RGB crop around the target (127x127, the exemplar size of the trackers) |
| `searches` | `(250, 271, 271, 3)` | `uint8` RGB search region belonging to the template (271x271, the SiamRPN search size). Regions that exceed the frame are padded with a gray color. |

To evaluate the patch, place [LaSOT](https://huggingface.co/datasets/l-lt/LaSOT)-format frame sequences under `data/<sequence-name>/img/00000001.jpg, 00000002.jpg, ...` (e.g. `data/person-7/img/`). The scripts default to `data/person-7/img` but accept any sequence via `--frames`. To reproduce the video sequence above, use the `person-7` datapoint from [LaSOT](https://huggingface.co/datasets/l-lt/LaSOT/blob/main/person.zip).

### 3. Vendor the third-party tracker implementations

```bash
git clone https://github.com/STVIR/pysot.git models/siamrpnpp/pysot_repo && mv models/siamrpnpp/pysot_repo/pysot models/siamrpnpp/pysot && rm -rf models/siamrpnpp/pysot_repo
git clone https://github.com/ohhhyeahhh/SiamCAR.git models/siamcar/siamcar_repo && mv models/siamcar/siamcar_repo/pysot models/siamcar/pysot && rm -rf models/siamcar/siamcar_repo
git clone https://github.com/hqucv/siamban.git models/siamban/siamban_repo && mv models/siamban/siamban_repo/siamban models/siamban/siamban && rm -rf models/siamban/siamban_repo
```

(DaSiamRPN is a self-contained port and needs no vendoring step. For SiamRPN, copy the upstream `siamrpn.py` as described in "Files not included".)

### 4. Download pretrained tracker weights

Download each checkpoint from its official source and place it as shown. `evaluate_video_siamrpnpp.py` and `evaluate_video_siamban.py` also need each variant's small `config*.yaml`, already included in this repo under the matching `models/<tracker>/` folder. The SiamCAR `config.yaml` is not included (see "Files not included").

| Tracker / variant | Place at | Source |
|---|---|---|
| SiamRPN | `models/siamrpn/model.pth` | `model.pth` in [huanglianghua/siamrpn-pytorch](https://github.com/huanglianghua/siamrpn-pytorch) |
| DaSiamRPN | `models/dasiamrpn/SiamRPNBIG.model` | [foolwood/DaSiamRPN](https://github.com/foolwood/DaSiamRPN) |
| SiamRPN++ ResNet50 | `models/siamrpnpp/model_resnet50.pth` | `siamrpn_r50_l234_dwxcorr` in [pysot MODEL_ZOO](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md) |
| SiamRPN++ AlexNet | `models/siamrpnpp/model_alexnet.pth` | `siamrpn_alex_dwxcorr` in [pysot MODEL_ZOO](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md) |
| SiamRPN++ MobileNetV2 | `models/siamrpnpp/model_mobilenet.pth` | `siamrpn_mobilev2_l234_dwxcorr` in [pysot MODEL_ZOO](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md) |
| SiamBAN | `models/siamban/model.pth` | `siamban_r50_l234` in [hqucv/siamban MODEL_ZOO](https://github.com/hqucv/siamban/blob/master/MODEL_ZOO.md) |
| SiamCAR (general) | `models/siamcar/model_general.pth` | `general_model` in [ohhhyeahhh/SiamCAR](https://github.com/ohhhyeahhh/SiamCAR) |
| SiamCAR (GOT-10k) | `models/siamcar/model_got10k.pth` | `got10k_model` in [ohhhyeahhh/SiamCAR](https://github.com/ohhhyeahhh/SiamCAR) |
| SiamCAR (LaSOT) | `models/siamcar/model_lasot.pth` | `LaSOT_model` in [ohhhyeahhh/SiamCAR](https://github.com/ohhhyeahhh/SiamCAR) |

### 5. Train a patch

```bash
python create_patch.py --data data/train.npz --epochs 100 --lr 0.005
```

### 6. Run a transferability evaluation

```bash
# DaSiamRPN
python evaluate_video_dasiamrpn.py --patch patch/patch_best.pt --frames data/person-7/img --init_x 557 --init_y 330 --init_w 83 --init_h 202

# SiamRPN++
python evaluate_video_siamrpnpp.py --patch patch/patch_best.pt --frames data/person-7/img --init_x 557 --init_y 330 --init_w 83 --init_h 202 --backbone resnet50

# SiamBAN
python evaluate_video_siamban.py --patch patch/patch_best.pt --frames data/person-7/img --init_x 557 --init_y 330 --init_w 83 --init_h 202

# SiamCAR LaSOT trained variant (needs the missing script)
python evaluate_video_siamcar.py --patch patch/patch_best.pt --frames data/person-7/img --init_x 557 --init_y 330 --init_w 83 --init_h 202 --variant lasot
```

Each script writes two videos under `results/<tracker>/video/<sequence>/`: the tracker's own search-crop view (`search_images/tracking.mp4`) and the patch composited into the full original frame (`frames/tracking_frame.mp4`).

### 7. Test a printed patch in the real world

```bash
python evaluate_physical_attack.py --tracker siamrpn --record results/physical/siamrpn.mp4 --record_raw results/physical/raw.mp4 --log results/physical/siamrpn.csv
```

Draw the initial box around the target with the mouse on the first frame and confirm with ENTER or SPACE. Press `r` to select a new box and `q` to quit.

- `--tracker` selects one of `siamrpn`, `dasiamrpn`, `siamrpnpp-resnet50`, `siamrpnpp-alexnet`, `siamrpnpp-mobilenet`, `siamban`, `siamcar-general`, `siamcar-lasot`, `siamcar-got10k`. `siamrpn` and the `siamcar-*` variants need the files listed under "Files not included".
- `--source` accepts a webcam index (default `0`), a video file or an image sequence.
- `--record_raw` saves the unannotated camera stream. Replay it with `--source results/physical/raw.mp4 --init_box X Y W H` to compare all trackers on identical footage.
- `--log` writes the box of every frame to a csv file.

No patch is pasted digitally here. The printed patch is part of the camera image. Keep in mind that the patch was trained on digital data of a full-body person and without a printability term, so its size relative to the target, the print quality and the lighting influence the result.

## To Be Added:
- **Footage of Physical Adversarial Patch Attack**.
- **Patch Training Extension**, e.g. using SiamRPN++ as source model to create a more effective patch.
- **Detection metrics**, e.g. mAP.
- **More augmentation techniques during training**, e.g. sheering, compression, lighting.
- **Include non-Siamese trackers** to test transferability across different architectures.
