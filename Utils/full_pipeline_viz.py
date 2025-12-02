import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
})

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Utils.dataset_loader import preprocess_mri 
from Utils.visualization import _to_uint8_image     


def contour_binary(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(out, contours, -1, 1, thickness=1)
    return out

def postprocess_mask(prob, thr=0.5, min_area=50):
    binm = (prob >= thr).astype(np.uint8)
    if binm.sum() == 0:
        return binm
    cnts, _ = cv2.findContours((binm*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(binm)
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 1, -1)
    return out

def compute_bbox(mask, pad, H, W):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        cx, cy = W//2, H//2
        half = min(H, W)//4
        x0 = max(0, cx-half); x1 = min(W-1, cx+half)
        y0 = max(0, cy-half); y1 = min(H-1, cy+half)
        return int(x0), int(y0), int(x1), int(y1)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W-1, x1); y1 = min(H-1, y1)
    return int(x0), int(y0), int(x1), int(y1)

def make_overlay(img, gt_mask, pred_mask):
    base = _to_uint8_image(img)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    gt_b = (gt_mask > 0).astype(np.uint8)
    pred_b = (pred_mask > 0).astype(np.uint8)

    gt_cont = contour_binary(gt_b)
    pred_cont = contour_binary(pred_b)

    overlay = base.copy()
    cnts, _ = cv2.findContours((gt_cont * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 215, 0), 2)  # yellow

    cnts, _ = cv2.findContours((pred_cont * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)  # cyan

    fp = (pred_b == 1) & (gt_b == 0)
    fn = (pred_b == 0) & (gt_b == 1)
    diff = np.zeros_like(overlay, dtype=np.uint8)
    diff[fp] = (220, 20, 20)    # red
    diff[fn] = (30, 144, 255)   # blue
    if fp.any() or fn.any():
        overlay = cv2.addWeighted(overlay, 1.0, diff, 0.45, 0)
    return overlay

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

def load_raw(path):
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32)
    else:
        # keep grayscale reading (dataset is grayscale)
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read: {path}")
        return img.astype(np.float32)


def visualize_pipeline(input_img, seg_model_path, clf_model_path, out_path, gt_mask_path=None, threshold=0.5, pad=16, min_area=50):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    meta = {"input": str(input_img), "seg_model": str(seg_model_path),
            "clf_model": str(clf_model_path), "timestamp": datetime.now().isoformat(),
            "threshold": float(threshold), "pad": int(pad)}

    print("🔹 Loading raw image...")
    raw = load_raw(input_img)                      # float32
    raw_uint8 = _to_uint8_image(raw)               # uint8 for viz
    H0, W0 = raw.shape[:2]

    seg_pp = preprocess_mri(raw)                  
    seg_pp = seg_pp / 255.0                        
    seg_input = np.expand_dims(seg_pp, axis=(0, -1))  
    seg_model = load_model(seg_model_path, compile=False)
    model_input_shape = seg_model.input_shape
    desired_c = 1
    if isinstance(model_input_shape, (list, tuple)) and len(model_input_shape) >= 4:
        desired_c = int(model_input_shape[-1])
    if seg_input.shape[-1] != desired_c:
        if desired_c == 3:
            seg_input = np.repeat(seg_input, 3, axis=-1)
        else:
            seg_input = seg_input[..., :desired_c]

    print("🔹 Running segmentation inference...")
    with tf.device(DEVICE):
        seg_out = seg_model.predict(seg_input, verbose=0)[0]
    prob_map_small = np.squeeze(seg_out)   # (H,W) or (H,W,C) handled below

    if prob_map_small.ndim == 3:
        if prob_map_small.shape[-1] == 1:
            prob_map_small = prob_map_small[..., 0]
        else:
            # pick max across non-background channels (assume channel 0 is background)
            if prob_map_small.shape[-1] >= 2:
                prob_map_small = np.max(prob_map_small[..., 1:], axis=-1)
            else:
                prob_map_small = prob_map_small[..., 0]

    prob_map = cv2.resize(prob_map_small.astype(np.float32), (W0, H0), interpolation=cv2.INTER_LINEAR)
    prob_map = np.clip(prob_map, 0.0, 1.0)
    pred_mask = postprocess_mask(prob_map, thr=threshold, min_area=min_area).astype(np.uint8)

    if gt_mask_path:
        gt = np.load(gt_mask_path).astype(np.uint8)
        gt_resized = cv2.resize(gt, (W0, H0), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt_resized > 0).astype(np.uint8)
    else:
        gt_bin = np.zeros_like(pred_mask)

    overlay_img = make_overlay(raw_uint8, gt_bin, pred_mask)
    x0, y0, x1, y1 = compute_bbox(pred_mask, pad, H0, W0)
    crop_raw = raw[y0:y1+1, x0:x1+1]
    crop_uint8 = _to_uint8_image(crop_raw)
    if crop_uint8.ndim == 3 and crop_uint8.shape[2] == 3:
        crop_vis = cv2.cvtColor(crop_uint8, cv2.COLOR_RGB2GRAY)
    else:
        crop_vis = crop_uint8

    print("🔹 Preprocessing crop for classifier...")
    crop_pp = preprocess_mri(crop_raw)         
    crop_pp = crop_pp / 255.0                   
    clf_input = np.expand_dims(crop_pp, axis=(0, -1))  

    clf_model = load_model(clf_model_path, compile=False)
    clf_input_shape = clf_model.input_shape
    desired_clf_c = 1
    if isinstance(clf_input_shape, (list, tuple)) and len(clf_input_shape) >= 4:
        desired_clf_c = int(clf_input_shape[-1])
    if clf_input.shape[-1] != desired_clf_c:
        if desired_clf_c == 3:
            clf_input = np.repeat(clf_input, 3, axis=-1)
        else:
            clf_input = clf_input[..., :desired_clf_c]

    print("🔹 Running classification inference...")
    with tf.device(DEVICE):
        class_probs = clf_model.predict(clf_input, verbose=0)[0]
    class_probs = np.clip(class_probs, 0.0, 1.0)
    pred_idx = int(np.argmax(class_probs))
    pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    stem = Path(input_img).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_path).parent if Path(out_path).suffix else Path(out_path)
    
    if not out_dir.exists():
        out_dir = Path(".")
    
    out_prefix = out_dir / f"{stem}_{timestamp}"
    mask_path = str(out_prefix) + "_mask.png"
    cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))
    overlay_path = str(out_prefix) + "_overlay.png"
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    crop_path = str(out_prefix) + "_crop.png"
    
    if crop_uint8.ndim == 2:
        cv2.imwrite(crop_path, cv2.cvtColor(crop_uint8, cv2.COLOR_GRAY2BGR))
    else:
        cv2.imwrite(crop_path, cv2.cvtColor(crop_uint8, cv2.COLOR_RGB2BGR))
    
    prob_norm = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)
    prob_u8 = (prob_norm * 255).astype(np.uint8)
    prob_color = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    prob_path = str(out_prefix) + "_probmap.png"
    cv2.imwrite(prob_path, prob_color)

    meta = {
        "input": str(input_img),
        "seg_model": str(seg_model_path),
        "clf_model": str(clf_model_path),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "class_probs": [float(x) for x in class_probs.tolist()],
        "class_pred": pred_label,
        "files": {
            "mask": mask_path,
            "overlay": overlay_path,
            "crop": crop_path,
            "probmap": prob_path
        }
    }
    meta_path = str(out_prefix) + "_meta.json"
    with open(meta_path, "w") as jf:
        json.dump(meta, jf, indent=2)


    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=True)

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.05, 1.05, 1.1],
        width_ratios=[1, 1]
    )

    def label(ax, letter):
        ax.text(
            0.02, 0.02, letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="semibold",
            ha="left", va="bottom",
            color="white",
            bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.2")
        )

    # Row 1
    # A: Original
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(raw_uint8, cmap="gray")
    axA.set_title("Original Image", pad=6)
    axA.axis("off")
    label(axA, "A")

    # D: Overlay
    axD = fig.add_subplot(gs[0, 1])
    axD.imshow(overlay_img)
    axD.set_title("GT (yellow) · Pred (cyan)\nFP (red), FN (blue)", pad=4)
    axD.axis("off")
    label(axD, "B")    

    # Row 2
    # C: Predicted Mask
    axC = fig.add_subplot(gs[1, 0])
    axC.imshow(pred_mask, cmap="gray")
    axC.set_title("Predicted Mask", pad=6)
    axC.axis("off")
    label(axC, "C")

    # Probability map
    axB = fig.add_subplot(gs[1, 1])
    im = axB.imshow(prob_map, cmap="jet", vmin=0, vmax=1)
    axB.set_title("Segmentation Probability", pad=6)
    axB.axis("off")
    label(axB, "D")
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(axB, width="70%", height="6%", loc="lower center", borderpad=1.4)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)

    # Row 3 
    # E: Cropped ROI
    axE = fig.add_subplot(gs[2, 0])
    crop_vis_show = cv2.resize(crop_vis, (300, 300), interpolation=cv2.INTER_AREA)
    axE.imshow(crop_vis_show, cmap="gray")
    axE.set_title("Cropped ROI", pad=6)
    axE.axis("off")
    label(axE, "E")

    
    gt_label = None

    if gt_mask_path is not None:
        label_path = str(gt_mask_path).replace("_mask.npy", "_label.npy")

        if os.path.exists(label_path):
            gt_raw = np.load(label_path)
            gt_idx = int(np.round(gt_raw.flatten()[0])) - 1  # ensure 0-based labels

            gt_label = CLASS_NAMES[gt_idx]

    # F: Classifier output
    axF = fig.add_subplot(gs[2, 1])

    cls = CLASS_NAMES
    probs = class_probs.tolist()
    ypos = np.arange(len(cls))

    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    gt_idx = None
    if 'gt_label' in locals() and gt_label in CLASS_NAMES:
        gt_idx = CLASS_NAMES.index(gt_label)

    colors = ["#CFE2F3"] * len(cls)  # soft grey-blue base
    bars_pred = axF.barh(
        ypos,
        probs,
        height=0.45,
        color=colors,
        edgecolor="none",
        label="Predicted"
    )

    bars_pred[pred_idx].set_color("#4C72B0")

    if gt_idx is not None:
        bar_gt = axF.barh(
            ypos[gt_idx],
            1.0,
            height=0.45,
            facecolor="none",
            edgecolor="#E74C3C",
            linewidth=2.0,
            label="Ground Truth"
        )

    for i, bar in enumerate(bars_pred):
        val = probs[i]
        axF.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height()/2,
            f"{val:.2f}",
            va="center",
            fontsize=10
        )

    axF.set_yticks(ypos)
    axF.set_yticklabels(cls)
    axF.set_xlim(0, 1.05)
    axF.set_xlabel("Probability")
    axF.set_title("GT vs Predicted Classification", pad=6)
    axF.legend(loc="upper left", frameon=False, fontsize=9)

    axF.grid(axis="x", linestyle="--", alpha=0.15)

    label(axF, "F")

    if Path(out_path).suffix:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print("Saved visualization:", out_path)
    else:
        default_out = str(out_prefix) + "_pipeline.png"
        fig.savefig(default_out, dpi=300, bbox_inches="tight")
        print("Saved visualization:", default_out)

    plt.close(fig)

    print("Done. Meta saved to:", meta_path)
    return meta_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input .npy or image")
    p.add_argument("--seg", required=True, help="Segmentation model path (.keras)")
    p.add_argument("--clf", required=True, help="Classifier model path (.keras)")
    p.add_argument("--gt_mask", default=None, help="Optional ground-truth mask (.npy)")
    p.add_argument("--out", default="pipeline_output.png", help="Output visualization path (PNG)")
    p.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    p.add_argument("--pad", type=int, default=16, help="Padding (px) around bbox")
    p.add_argument("--min_area", type=int, default=50, help="Min area to keep segmented component")
    args = p.parse_args()

    visualize_pipeline(
        args.input, args.seg, args.clf, args.out,
        gt_mask_path=args.gt_mask, threshold=args.threshold, pad=args.pad, min_area=args.min_area
    )
