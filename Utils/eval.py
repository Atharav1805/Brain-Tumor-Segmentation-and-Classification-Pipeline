import os
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.dataset_loader import load_figshare_npy_dataset
from Utils.visualization import plot_confusion_matrix, plot_grid, _to_uint8_image
from Utils.gradcam import compute_gradcam, get_last_conv_layer, save_gradcam_figure

# Config
DATA_DIR = "Data/converted_npy"
MODEL_PATH = "Outputs/custom_cnn/fold_1/best_model.keras"
OUTPUT_ROOT = "Evaluation_classifier"
CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]
BATCH_SIZE = 8
SEED = 42

ENABLE = {
    "gradcam": True,
    "mean_gradcam": True,
    "tsne": True,
    "calibration": True,
    "robustness": False,
    "learning_curves": True,
    "save_examples": True
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_bbox(mask, pad=16):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:    # fallback for missing masks
        h, w = mask.shape
        cx, cy = w//2, h//2
        r = min(h, w)//4
        return cx-r, cy-r, cx+r, cy+r

    x0, x1 = xs.min() - pad, xs.max() + pad
    y0, y1 = ys.min() - pad, ys.max() + pad
    return max(0,x0), max(0,y0), x1, y1

def crop_tumor(img, mask, out_size=256):
    if img.ndim == 3:
        img = img[:, :, 0]

    x0,y0,x1,y1 = safe_bbox(mask)
    crop = img[y0:y1, x0:x1]

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    crop = crop.astype(np.float32)

    if crop.max() > 1:
        crop /= 255.0

    return np.expand_dims(crop, axis=-1)    # (256,256,1)

def run_evaluation(data_dir, model_path, out_root):
    ensure_dir(out_root)
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    print("🔹 Loading dataset...")
    X_raw, y, masks, pids = load_figshare_npy_dataset(data_dir, preprocess=True)
    masks = np.array(masks)

    print(f"Loaded raw: {X_raw.shape}, masks: {masks.shape}")
    print("🔹 Cropping tumors (matching training)...")
    X = np.zeros((len(X_raw), 256, 256, 1), dtype=np.float32)
    for i in range(len(X)):
        X[i] = crop_tumor(X_raw[i], masks[i])
    print("Evaluating on", X.shape)

    print("🔹 Loading model...")
    model = load_model(model_path, compile=False)
    model.summary()

    print("🔹 Predicting...")
    t0 = time.time()
    y_probs = []
    for i in range(0, len(X), BATCH_SIZE):
        y_probs.append(model.predict(X[i:i+BATCH_SIZE], verbose=0))
    y_probs = np.vstack(y_probs)
    print("✔ Done in %.2fs" % (time.time() - t0))

    y_pred = np.argmax(y_probs, axis=1)
    n_classes = len(CLASS_NAMES)
    ensure_dir(out_root + "/plots")
    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix:\n", cm)
    plot_confusion_matrix(y, y_pred, CLASS_NAMES, save=f"{out_root}/plots/confusion_matrix.png")

    report = classification_report(y, y_pred, target_names=CLASS_NAMES, output_dict=True)
    save_json(report, f"{out_root}/classification_report.json")

    print("Classification Report:")
    print(json.dumps(report, indent=2))

    # Save per-class summary
    prec, rec, f1, sup = precision_recall_fscore_support(y, y_pred)
    class_summary = [
        {"class": CLASS_NAMES[i], "precision": float(prec[i]), "recall": float(rec[i]),
         "f1": float(f1[i]), "support": int(sup[i])}
        for i in range(n_classes)
    ]
    save_json(class_summary, f"{out_root}/per_class_metrics.json")

    print("🔹 Plotting ROC & PR curves...")
    y_bin = label_binarize(y, classes=np.arange(n_classes))

    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_probs[:,i])
        aucv = auc(fpr,tpr)
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={aucv:.3f})")
    plt.plot([0,1],[0,1],"k--", alpha=0.5)
    plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.savefig(f"{out_root}/plots/roc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        pr, rc, _ = precision_recall_curve(y_bin[:,i], y_probs[:,i])
        ap = auc(rc, pr)
        plt.plot(rc, pr, label=f"{CLASS_NAMES[i]} (AP={ap:.3f})")
    plt.legend(); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.savefig(f"{out_root}/plots/pr.png", dpi=150)
    plt.close()

    if ENABLE["calibration"]:
        print("🔹 Calibration curves...")
        plt.figure(figsize=(7,6))
        for i in range(n_classes):
            prob_true, prob_pred = calibration_curve(y_bin[:,i], y_probs[:,i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=CLASS_NAMES[i])
        plt.plot([0,1],[0,1],"k--")
        plt.legend(); plt.title("Calibration curves")
        plt.savefig(f"{out_root}/plots/calibration.png", dpi=150)
        plt.close()

    if ENABLE["gradcam"]:
        print("🔹 Grad-CAM...")
        last_conv = get_last_conv_layer(model)
        gdir = f"{out_root}/gradcam"
        ensure_dir(gdir)

        chosen = random.sample(range(len(X)), min(12, len(X)))

        grads_by_class = {i: [] for i in range(n_classes)}
        imgs_by_class = {i: [] for i in range(n_classes)}

        for idx in chosen:
            img = X[idx]
            pred = y_pred[idx]
            true = y[idx]

            heatmap = compute_gradcam(model, img, last_conv, class_index=pred)
            grads_by_class[true].append(heatmap)
            imgs_by_class[true].append(img)

            out = f"{gdir}/sample_{idx}_true{true}_pred{pred}.png"
            title = f"Grad-CAM | True {CLASS_NAMES[true]} | Pred {CLASS_NAMES[pred]}"
            save_gradcam_figure(img, heatmap, title, out)

        if ENABLE["mean_gradcam"]:
            mdir = f"{out_root}/mean_gradcam"
            ensure_dir(mdir)
            for cls in range(n_classes):
                if len(grads_by_class[cls]) == 0:
                    continue
                mean_hm = np.mean(np.stack(grads_by_class[cls],0),0)
                base_img = imgs_by_class[cls][0]
                save_gradcam_figure(base_img, mean_hm, f"Mean Grad-CAM: {CLASS_NAMES[cls]}", f"{mdir}/mean_gradcam_cls{cls}.png")

    if ENABLE["tsne"]:
        print("🔹 Running t-SNE...")
        penultimate = None
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 2:
                    penultimate = layer
                    break
            except:
                pass

        if penultimate is None:
            penultimate = model.layers[-2]

        feat_model = Model(inputs=model.input, outputs=penultimate.output)
        feats = feat_model.predict(X, batch_size=4)

        emb = TSNE(n_components=2, perplexity=40, random_state=SEED).fit_transform(feats)

        plt.figure(figsize=(8,6))
        for cls in range(n_classes):
            plt.scatter(
                emb[y==cls,0], emb[y==cls,1], s=8, alpha=0.7, label=CLASS_NAMES[cls]
            )
        mis = np.where(y != y_pred)[0]
        plt.scatter(
            emb[mis,0], emb[mis,1], facecolors='none', edgecolors='k',
            s=50, linewidths=0.7, label="Misclassified"
        )
        plt.legend(); plt.title("t-SNE features")
        plt.savefig(f"{out_root}/plots/tsne.png", dpi=150)
        plt.close()


    if ENABLE["save_examples"]:
        print("🔹 Saving examples...")
        edir = f"{out_root}/examples"
        ensure_dir(edir)

        correct = np.where(y == y_pred)[0][:16]
        wrong   = np.where(y != y_pred)[0][:16]

        if len(correct) > 0:
            imgs = [X[i] for i in correct]
            lbls = [CLASS_NAMES[y[i]] for i in correct]
            preds = [CLASS_NAMES[y_pred[i]] for i in correct]
            fig,_ = plot_grid(imgs, None, preds, lbls, ncols=4)
            fig.savefig(f"{edir}/correct.png", dpi=150); plt.close(fig)

        if len(wrong) > 0:
            imgs = [X[i] for i in wrong]
            lbls = [CLASS_NAMES[y[i]] for i in wrong]
            preds = [CLASS_NAMES[y_pred[i]] for i in wrong]
            fig,_ = plot_grid(imgs, None, preds, lbls, ncols=4)
            fig.savefig(f"{edir}/wrong.png", dpi=150); plt.close(fig)

    print("\nEvaluation complete. Outputs saved in:", out_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--out", default=OUTPUT_ROOT)
    args = parser.parse_args()
    run_evaluation(args.data_dir, args.model, args.out)
