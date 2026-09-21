import argparse
import csv
import os
import time

import cv2
import numpy as np


TRACKERS = ["siamrpn", "dasiamrpn", "siamrpnpp-resnet50", "siamrpnpp-alexnet", "siamrpnpp-mobilenet", "siamban", "siamcar-general", "siamcar-lasot", "siamcar-got10k"]


# Builds the selected tracker and returns two functions: init(frame, box) and update(frame) -> [x, y, w, h].
def build_tracker(name, weights, config):

    if name == "siamrpn":
        from models.siamrpn.siamrpn import TrackerSiamRPN
        tracker = TrackerSiamRPN(net_path=weights or "models/siamrpn/model.pth")
        return tracker.init, tracker.update

    if name == "dasiamrpn":
        from models.dasiamrpn.dasiamrpn import TrackerDaSiamRPN
        from evaluate_video_dasiamrpn import update_with_patch
        tracker = TrackerDaSiamRPN(net_path=weights or "models/dasiamrpn/SiamRPNBIG.model")

        # Without a patch, update_with_patch() is the plain DaSiamRPN update
        return tracker.init, lambda frame: update_with_patch(tracker, frame, None, 0, 0, 0, 0)[1]

    if name.startswith("siamrpnpp-"):
        from models.siamrpnpp.siamrpnpp import build_siamrpnpp_tracker
        backbone = name.split("-")[1]
        tracker = build_siamrpnpp_tracker(net_path=weights or f"models/siamrpnpp/model_{backbone}.pth", config_path=config or f"models/siamrpnpp/config_{backbone}.yaml")
        return tracker.init, lambda frame: tracker.track(frame)["bbox"]

    if name == "siamban":
        from models.siamban.siamban_tracker import build_siamban_tracker
        tracker = build_siamban_tracker(net_path=weights or "models/siamban/model.pth", config_path=config or "models/siamban/config.yaml")
        return tracker.init, lambda frame: tracker.track(frame)["bbox"]

    if name.startswith("siamcar-"):
        from models.siamcar.siamcar import build_siamcar_tracker
        variant = name.split("-")[1]
        tracker = build_siamcar_tracker(net_path=weights or f"models/siamcar/model_{variant}.pth", config_path=config or "models/siamcar/config.yaml")
        return tracker.init, lambda frame: tracker.track(frame, tracker.hp)["bbox"]

    raise ValueError(f"Unknown tracker '{name}', choose one of {TRACKERS}")


def select_box(window, frame):
    x, y, w, h = cv2.selectROI(window, frame, showCrosshair=False)
    return np.array([x, y, w, h], dtype=np.float32) if w > 0 and h > 0 else None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=TRACKERS, default="siamrpn")
    parser.add_argument("--weights", default=None, help="Overrides the default weights of the selected tracker.")
    parser.add_argument("--config", default=None, help="Overrides the default config of the selected tracker.")
    parser.add_argument("--source", default="0", help="Webcam index, video file or image sequence (e.g. data/person-7/img/%%08d.jpg).")
    parser.add_argument("--init_box", type=float, nargs=4, default=None, metavar=("X", "Y", "W", "H"), help="Initial box. If not given, it is drawn with the mouse on the first frame.")
    parser.add_argument("--record", default=None, help="Saves the annotated stream to this video file.")
    parser.add_argument("--record_raw", default=None, help="Saves the unannotated camera stream, to replay the same footage with other trackers via --source.")
    parser.add_argument("--log", default=None, help="Saves the box of every frame to this csv file.")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)

    args = parser.parse_args()
    window = "Physical attack - " + args.tracker

    # cv2.VideoWriter fails silently if the output folder does not exist, so create all of them up front
    for path in (args.record, args.record_raw, args.log):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    init, update = build_tracker(args.tracker, args.weights, args.config)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read from source '{args.source}'")

    # Initial box: drawn with the mouse on the first frame (ENTER/SPACE confirms, c cancels)
    box = np.array(args.init_box, dtype=np.float32) if args.init_box is not None else select_box(window, frame)
    if box is None:
        return
    init(frame, box)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.record, fourcc, fps, (w, h)) if args.record else None
    writer_raw = cv2.VideoWriter(args.record_raw, fourcc, fps, (w, h)) if args.record_raw else None

    log_file = open(args.log, "w", newline="") if args.log else None
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(["frame", "x", "y", "w", "h", "cx", "cy"])

    i = 0
    while True:
        i += 1
        if writer_raw:
            writer_raw.write(frame)

        t0 = time.time()
        x, y, bw, bh = [float(v) for v in update(frame)]
        latency = time.time() - t0

        if log:
            log.writerow([i, x, y, bw, bh, x + bw / 2, y + bh / 2])

        annotated = frame.copy()
        cv2.rectangle(annotated, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 2)
        cv2.putText(annotated, f"{args.tracker}  {1 / max(latency, 1e-6):.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF

            # q: quit, r: select a new box (re-initializes the tracker on the current frame)
            if key == ord("q"):
                break
            if key == ord("r"):
                new_box = select_box(window, frame)
                if new_box is not None:
                    init(frame, new_box)

        if args.max_frames and i >= args.max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    for out in (writer, writer_raw, log_file):
        if out:
            out.release() if hasattr(out, "release") else out.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

    # Physical attack with a printed patch, live webcam, initial box drawn with the mouse:
    # python .\evaluate_physical_attack.py --record results/physical/siamrpn.mp4 --record_raw results/physical/raw.mp4 --log results/physical/siamrpn.csv --tracker siamrpn

    # Same footage with another tracker:
    # python .\evaluate_physical_attack.py --tracker siamban --source results/physical/raw.mp4 --init_box 200 100 150 200
