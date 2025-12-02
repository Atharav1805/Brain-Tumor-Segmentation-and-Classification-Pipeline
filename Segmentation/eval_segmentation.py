import os
import sys
import argparse
from pathlib import Path
import json
import csv
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


try:
    from Utils.dataset_loader import load_figshare_npy_dataset
    from postprocessing import postprocess_mask
    from Utils.visualization import save_prediction_visual, _to_uint8_image, _prepare_mask
except Exception as e:
    # fallback: try to import when running from repo root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from Utils.dataset_loader import load_figshare_npy_dataset
    from postprocessing import postprocess_mask
    from Utils.visualization import save_prediction_visual, _to_uint8_image, _prepare_mask


def dice_np(y_true, y_pred, eps=1e-8):
    y_true_f = y_true.astype(np.uint8).ravel()
    y_pred_f = y_pred.astype(np.uint8).ravel()
    inter = np.sum(y_true_f * y_pred_f)
    return float((2.0 * inter + eps) / (np.sum(y_true_f) + np.sum(y_pred_f) + eps))

def iou_np(y_true, y_pred, eps=1e-8):
    y_true_f = y_true.astype(np.uint8).ravel()
    y_pred_f = y_pred.astype(np.uint8).ravel()
    inter = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - inter
    return float((inter + eps) / (union + eps))

def pixel_accuracy(y_true, y_pred):
    return float(np.mean((y_true.astype(np.uint8).ravel() == y_pred.astype(np.uint8).ravel()).astype(np.float32)))

def sensitivity_np(y_true, y_pred, eps=1e-8):
    y_true_f = y_true.astype(np.uint8).ravel()
    y_pred_f = y_pred.astype(np.uint8).ravel()
    tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
    fn = np.sum((y_true_f == 1) & (y_pred_f == 0))
    return float((tp + eps) / (tp + fn + eps))

def specificity_np(y_true, y_pred, eps=1e-8):
    y_true_f = y_true.astype(np.uint8).ravel()
    y_pred_f = y_pred.astype(np.uint8).ravel()
    tn = np.sum((y_true_f == 0) & (y_pred_f == 0))
    fp = np.sum((y_true_f == 0) & (y_pred_f == 1))
    return float((tn + eps) / (tn + fp + eps))

def precision_np(y_true, y_pred, eps=1e-8):
    y_true_f = y_true.astype(np.uint8).ravel()
    y_pred_f = y_pred.astype(np.uint8).ravel()
    tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
    fp = np.sum((y_true_f == 0) & (y_pred_f == 1))
    return float((tp + eps) / (tp + fp + eps))

def f1_np(y_true, y_pred, eps=1e-8):
    p = precision_np(y_true, y_pred, eps)
    r = sensitivity_np(y_true, y_pred, eps)
    return float((2 * p * r + eps) / (p + r + eps))


def contour_binary(mask):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    canvas = np.zeros_like(mask_u8)
    cv2.drawContours(canvas, contours, -1, 255, 1)  # thickness=1
    return (canvas > 0).astype(np.uint8)

def distance_map(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    inv = 1 - mask_u8
    dist_bg = cv2.distanceTransform((inv * 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    dist_fg = cv2.distanceTransform((mask_u8 * 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    
    # Combine into a symmetric map
    combined = dist_bg
    combined[mask_u8 == 1] = dist_fg[mask_u8 == 1]
    return combined

def boundary_error_heatmap(gt_mask, pred_mask):
    dm_gt = distance_map(gt_mask)
    dm_pred = distance_map(pred_mask)
    return np.abs(dm_gt - dm_pred)


def make_overlay_image(img_rgb, gt_mask, pred_mask, alpha=0.5):
    base = _to_uint8_image(img_rgb) 
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    h, w = base.shape[:2]
    gt_b = (gt_mask > 0).astype(np.uint8)
    pred_b = (pred_mask > 0).astype(np.uint8)

    gt_cont = contour_binary(gt_b)
    pred_cont = contour_binary(pred_b)

    overlay = base.copy()

    contours, _ = cv2.findContours((gt_cont * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 215, 0), 1) 

    contours, _ = cv2.findContours((pred_cont * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1) 

    # Difference map: FP red, FN blue
    fp = (pred_b == 1) & (gt_b == 0)
    fn = (pred_b == 0) & (gt_b == 1)
    diff = np.zeros_like(overlay, dtype=np.uint8)
    diff[fp] = (220, 20, 20)   # red
    diff[fn] = (30, 144, 255)  # blue

    mask_diff = (fp | fn).astype(np.uint8)
    if mask_diff.any():
        diff_alpha = 0.45
        overlay = cv2.addWeighted(overlay, 1.0, diff, diff_alpha, 0)

    return overlay

def get_last_conv_layer(model): # return the name of the last Conv2D layer in U-Net
    for layer in reversed(model.layers):
        try:
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        except:
            pass
    raise ValueError("No Conv2D layer found in model.")


def compute_seg_gradcam(model, img_gray, last_conv):
    img_inp = np.expand_dims(img_gray, 0)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_inp)
        # We take gradient wrt tumor probability
        class_channel = pred[..., 0]   # tumor channel
    grads = tape.gradient(class_channel, conv_out)  

    # channel-wise weights
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))  
    conv_out = conv_out[0] 

    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= (heatmap.max() + 1e-8)
    return heatmap


def draw_gradcam_overlay(img_rgb, heatmap, alpha=0.45):
    base = _to_uint8_image(img_rgb)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    h, w = base.shape[:2]
    hm_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_color = cv2.resize(hm_color, (w, h))

    overlay = cv2.addWeighted(base, 1 - alpha, hm_color, alpha, 0)
    return overlay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="Data/converted_npy", help="Dataset folder (contains *_image.npy etc.)")
    p.add_argument("--model", default="Outputs/UNet_preprocessed/best_unet.keras", help="Trained U-Net model (.keras)")
    p.add_argument("--out", default="Outputs/UNet_preprocessed/eval", help="Evaluation output root")
    p.add_argument("--preprocess", default="True", help="Load preprocessed images (True/False)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--samples", type=int, default=15, help="Number of sample overlays to save")
    p.add_argument("--random_seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    DATA_DIR = args.data_dir
    MODEL_PATH = args.model
    OUT_ROOT = args.out
    PREPROCESS = str(args.preprocess).lower() in ("1", "true", "yes")
    BATCH = int(args.batch_size)
    THRESH = float(args.threshold)
    N_SAMPLES = int(args.samples)
    SEED = int(args.random_seed)

    os.makedirs(OUT_ROOT, exist_ok=True)
    plots_dir = os.path.join(OUT_ROOT, "plots")
    samples_dir = os.path.join(OUT_ROOT, "samples")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    print("🔹 Loading dataset...")
    X, y, masks, pids = load_figshare_npy_dataset(DATA_DIR, preprocess=PREPROCESS)
    X = np.asarray(X, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.uint8)
    n = len(X)
    print(f"Loaded {n} samples. X shape: {X.shape}")

    X_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X], dtype=np.float32)
    X_gray = X_gray / 255.0
    X_gray = np.expand_dims(X_gray, -1)

    print("🔹 Loading model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    preds_prob_list = []
    for i in range(0, n, BATCH):
        batch = X_gray[i:i+BATCH]
        preds_prob_list.append(model.predict(batch, verbose=0))
    preds_prob = np.vstack(preds_prob_list)
    if preds_prob.ndim == 4 and preds_prob.shape[-1] == 1:
        preds_prob = preds_prob[..., 0]
    print("Predictions shape:", preds_prob.shape)

    rng = np.random.RandomState(SEED)
    save_idx = set(rng.choice(np.arange(n), size=min(N_SAMPLES, n), replace=False))

    per_sample_metrics = []
    dice_list, iou_list, acc_list, sens_list, spec_list, prec_list, f1_list = ([] for _ in range(7))

    print("🔹 Evaluating samples...")
    for idx in tqdm(range(n)):
        prob_map = preds_prob[idx]
        raw_bin = (prob_map >= THRESH).astype(np.uint8)
        try:
            post_bin = postprocess_mask(prob_map)
        except Exception:
            post_bin = postprocess_mask(raw_bin)
        post_bin = (post_bin > 0).astype(np.uint8)

        gt_mask = masks[idx]
        if gt_mask.ndim == 3:
            gt_mask = np.squeeze(gt_mask, axis=-1)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        d = dice_np(gt_mask, post_bin)
        j = iou_np(gt_mask, post_bin)
        acc = pixel_accuracy(gt_mask, post_bin)
        sens = sensitivity_np(gt_mask, post_bin)
        spec = specificity_np(gt_mask, post_bin)
        prec = precision_np(gt_mask, post_bin)
        f1 = f1_np(gt_mask, post_bin)

        per_sample_metrics.append({
            "idx": int(idx),
            "pid": str(pids[idx]) if pids is not None else None,
            "dice": d,
            "iou": j,
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "f1": f1
        })

        dice_list.append(d); iou_list.append(j); acc_list.append(acc)
        sens_list.append(sens); spec_list.append(spec); prec_list.append(prec); f1_list.append(f1)

        if idx in save_idx:
            img_rgb = X[idx]  # preprocessed RGB image from loader
            overlay = make_overlay_image(img_rgb, gt_mask, post_bin)
            overlay_path = os.path.join(samples_dir, f"sample_{idx:04d}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            be_map = boundary_error_heatmap(gt_mask, post_bin)
            bm_vis = (be_map / (be_map.max() + 1e-8) * 255).astype(np.uint8)
            bm_vis = cv2.applyColorMap(bm_vis, cv2.COLORMAP_JET)
            diff_path = os.path.join(samples_dir, f"sample_{idx:04d}_boundary_error.png")
            cv2.imwrite(diff_path, bm_vis)

            try:
                last_conv = get_last_conv_layer(model)
                heatmap = compute_seg_gradcam(model, X_gray[idx], last_conv)
                gradcam_overlay = draw_gradcam_overlay(img_rgb, heatmap)
                gc_path = os.path.join(samples_dir, f"sample_{idx:04d}_gradcam.png")
                cv2.imwrite(gc_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"[GradCAM error @ idx {idx}]: {e}")


    # Aggregate metrics
    dice_arr = np.array(dice_list)
    iou_arr = np.array(iou_list)
    acc_arr = np.array(acc_list)
    sens_arr = np.array(sens_list)
    spec_arr = np.array(spec_list)
    prec_arr = np.array(prec_list)
    f1_arr = np.array(f1_list)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(n),
        "dice_mean": float(np.mean(dice_arr)),
        "dice_std": float(np.std(dice_arr)),
        "dice_min": float(np.min(dice_arr)),
        "dice_max": float(np.max(dice_arr)),
        "iou_mean": float(np.mean(iou_arr)),
        "iou_std": float(np.std(iou_arr)),
        "accuracy_mean": float(np.mean(acc_arr)),
        "accuracy_std": float(np.std(acc_arr)),
        "sensitivity_mean": float(np.mean(sens_arr)),
        "sensitivity_std": float(np.std(sens_arr)),
        "specificity_mean": float(np.mean(spec_arr)),
        "specificity_std": float(np.std(spec_arr)),
        "precision_mean": float(np.mean(prec_arr)),
        "precision_std": float(np.std(prec_arr)),
        "f1_mean": float(np.mean(f1_arr)),
        "f1_std": float(np.std(f1_arr)),
    }

    csv_path = os.path.join(OUT_ROOT, "per_sample_metrics.csv")
    json_path = os.path.join(OUT_ROOT, "per_sample_metrics.json")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(per_sample_metrics[0].keys()))
        writer.writeheader()
        for row in per_sample_metrics:
            writer.writerow(row)
    with open(json_path, "w") as jf:
        json.dump(per_sample_metrics, jf, indent=2)

    agg_json = os.path.join(OUT_ROOT, "aggregate_metrics.json")
    with open(agg_json, "w") as af:
        json.dump(summary, af, indent=2)

    agg_txt = os.path.join(OUT_ROOT, "aggregate_metrics.txt")
    with open(agg_txt, "w") as lf:
        lf.write("Segmentation evaluation summary\n")
        lf.write(json.dumps(summary, indent=2))

    print("🔹 Generating plots...")
    plt.figure(figsize=(6,4))
    sns.histplot(dice_arr, bins=30, kde=True)
    plt.xlabel("Dice")
    plt.title("Dice distribution")
    plt.savefig(os.path.join(plots_dir, "dice_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(iou_arr, bins=30, kde=True)
    plt.xlabel("IoU")
    plt.title("IoU distribution")
    plt.savefig(os.path.join(plots_dir, "iou_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6,4))
    sns.boxplot(data=[dice_arr, iou_arr], orient="v")
    plt.xticks([0,1], ["Dice", "IoU"])
    plt.title("Metrics Boxplot")
    plt.savefig(os.path.join(plots_dir, "metrics_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("🔹 Computing pixel-level confusion matrix (this may be large)...")
    all_gt_flat = []
    all_pred_flat = []
    for idx in range(n):
        gt_mask = masks[idx]
        if gt_mask.ndim == 3:
            gt_mask = np.squeeze(gt_mask, axis=-1)
        gt_b = (gt_mask > 0).astype(np.uint8)
        pred_b = (postprocess_mask(preds_prob[idx]) > 0).astype(np.uint8)
        all_gt_flat.append(gt_b.ravel())
        all_pred_flat.append(pred_b.ravel())
    all_gt_flat = np.concatenate(all_gt_flat)
    all_pred_flat = np.concatenate(all_pred_flat)
    cm = confusion_matrix(all_gt_flat, all_pred_flat)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Pixel Confusion Matrix")
    plt.savefig(os.path.join(plots_dir, "pixel_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("🔹 Computing mean boundary error map (may be slow)...")
    be_acc = None
    for idx in range(n):
        gt_mask = masks[idx]
        if gt_mask.ndim == 3:
            gt_mask = np.squeeze(gt_mask, axis=-1)
        pred_mask = (postprocess_mask(preds_prob[idx]) > 0).astype(np.uint8)
        bemap = boundary_error_heatmap(gt_mask, pred_mask)
        if be_acc is None:
            be_acc = np.zeros_like(bemap, dtype=np.float32)
        be_acc += bemap
    be_mean = be_acc / float(n)
    be_vis = (be_mean / (be_mean.max() + 1e-8) * 255).astype(np.uint8)
    be_vis = cv2.applyColorMap(be_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(plots_dir, "mean_boundary_error.png"), be_vis)
    


    hist_dir = Path(MODEL_PATH).parent / "history"
    if hist_dir.exists():
        history_files = list(hist_dir.glob("*.npy"))
        for hf in history_files:
            try:
                h = np.load(str(hf), allow_pickle=True).item()
            except Exception:
                continue
            plt.figure(figsize=(6,4))
            if "loss" in h:
                plt.plot(h.get("loss", []), label="train_loss")
            if "val_loss" in h:
                plt.plot(h.get("val_loss", []), label="val_loss")
            plt.legend(); plt.title(f"Loss ({hf.stem})")
            plt.savefig(os.path.join(plots_dir, f"loss_{hf.stem}.png"), dpi=150, bbox_inches='tight')
            plt.close()

            if "dice_coef" in h or "val_dice_coef" in h:
                plt.figure(figsize=(6,4))
                if "dice_coef" in h:
                    plt.plot(h.get("dice_coef", []), label="train_dice")
                if "val_dice_coef" in h:
                    plt.plot(h.get("val_dice_coef", []), label="val_dice")
                plt.legend(); plt.title(f"Dice ({hf.stem})")
                plt.savefig(os.path.join(plots_dir, f"dice_{hf.stem}.png"), dpi=150, bbox_inches='tight')
                plt.close()

    print("✅ Segmentation evaluation complete.")
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("Files saved under:", OUT_ROOT)

if __name__ == "__main__":
    main()