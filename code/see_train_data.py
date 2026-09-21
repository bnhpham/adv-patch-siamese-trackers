import argparse
import os

import cv2
import numpy as np


# Saves every template/search pair of train.npz as one JPG (template on the left, search image on the right)
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.npz")
    parser.add_argument("--out_dir", default="data/train_samples")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.data)
    templates, searches = data["templates"], data["searches"]
    angles = data["angles"] if "angles" in data.files else None
    print({key: (data[key].shape, data[key].dtype) for key in data.files})

    header = 30
    for i in range(len(templates)):
        z, x = templates[i], searches[i]

        # Both images are stored as RGB, cv2 writes BGR
        canvas = np.zeros((header + x.shape[0], z.shape[1] + x.shape[1], 3), dtype=np.uint8)
        canvas[header:header + z.shape[0], :z.shape[1]] = z[..., ::-1]
        canvas[header:, z.shape[1]:] = x[..., ::-1]

        label = f"sample {i:03d}" + (f"  angle {int(angles[i])}" if angles is not None else "")
        cv2.putText(canvas, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.out_dir, f"{i:03d}.jpg"), canvas)

    print(f"saved {len(templates)} samples to {args.out_dir}")


if __name__ == "__main__":
    main()

    # python .\see_train_data.py --data data/train.npz --out_dir data/train_samples
