import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))
from Utils.dataset_loader import load_figshare_npy_dataset, preprocess_mri

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

def add_noise(x, sigma):
    noisy = x + np.random.normal(0, sigma, x.shape)
    return noisy


def compute_bbox_from_gt(mask, pad=12):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:   # if GT somehow empty → fallback center crop
        h, w = mask.shape
        cx, cy = w//2, h//2
        half = min(h, w)//4
        return cx-half, cy-half, cx+half, cy+half

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    x0 -= pad; y0 -= pad
    x1 += pad; y1 += pad

    h, w = mask.shape
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w-1, x1); y1 = min(h-1, y1)

    return x0, y0, x1, y1

def evaluate_noise_curve(data_dir, clf_path, out_path):

    print("🔹 Loading dataset...")
    X_raw, y, masks, _ = load_figshare_npy_dataset(data_dir, preprocess=True)

    print("Dataset:", X_raw.shape, y.shape)

    print("🔹 Loading classifier...", clf_path)
    clf = tf.keras.models.load_model(clf_path, compile=False)

    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.20]
    accuracies = []

    print("🔹 Running evaluation using GT cropping...")
    with tf.device(DEVICE):

        for sigma in noise_levels:
            print(f"\n→ Testing noise σ = {sigma}")

            preds = []

            for i in range(len(X_raw)):
                raw = X_raw[i]      
                gt_mask = masks[i]   

                # Add Gaussian noise to RAW
                noisy = add_noise(raw, sigma)
                noisy = np.clip(noisy, raw.min(), raw.max())

                # Crop using GT mask
                x0, y0, x1, y1 = compute_bbox_from_gt(gt_mask)
                crop = noisy[y0:y1+1, x0:x1+1]

                pp = cv2.CAP_PROP_APERTURE / 255.0  

                inp = np.expand_dims(pp, axis=(0, -1))  # (1,256,256,1)

                # Classify
                prob = clf.predict(inp, verbose=0)[0]
                pred = int(np.argmax(prob))

                preds.append(pred)

            preds = np.array(preds)
            acc = (preds == y).mean()
            accuracies.append(acc)

            print(f"   Accuracy = {acc:.4f}")


    plt.figure(figsize=(7,5))
    plt.plot(noise_levels, accuracies, marker='o', linewidth=2)
    plt.title("Classifier Robustness to Gaussian Noise (GT Cropping)")
    plt.xlabel("Noise Standard Deviation σ")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n✅ Saved noise robustness plot →", out_path)
    print("Noise levels:", noise_levels)
    print("Accuracies:", accuracies)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--clf", required=True)
    p.add_argument("--out", default="noise_vs_accuracy_gtcrop.png")
    args = p.parse_args()

    evaluate_noise_curve(args.data, args.clf, args.out)
