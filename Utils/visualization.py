from typing import Optional, Tuple, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    from skimage import measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

def _to_uint8_image(img: np.ndarray, clip: bool = True) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    if img.dtype == np.uint8:
        return img.copy()
    img = np.asarray(img).astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        p1, p99 = np.percentile(lum, (1.0, 99.0))
    else:
        p1, p99 = np.percentile(img, (1.0, 99.0))
    if p99 - p1 <= 0:
        mn, mx = float(img.min()), float(img.max())
    else:
        mn, mx = float(p1), float(p99)
    out = (img - mn) / (mx - mn + 1e-8)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    out = (255.0 * out).astype(np.uint8)
    if out.ndim == 2:
        return out
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    return out

def _prepare_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.dtype == bool:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        return mask > 0.5
    return mask.astype(np.int32) != 0

def show_image_and_mask(img: np.ndarray, mask: Optional[np.ndarray] = None, label_name: str = "", pid: Optional[str] = None, show_contour: bool = True, alpha: float = 0.35, figsize: Tuple[int,int] = (10,5), save: Optional[str] = None):
    img_u8 = _to_uint8_image(img)
    mask_bool = _prepare_mask(mask)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(img_u8, cmap='gray')
    axes[0].set_title("Original MRI")
    axes[0].axis('off')

    axes[1].imshow(img_u8, cmap='gray')
    if mask_bool is not None:
        if show_contour and _HAS_SKIMAGE:
            contours = measure.find_contours(mask_bool.astype(np.uint8), 0.5)
            axes[1].imshow(img_u8, cmap='gray')
            for contour in contours:
                axes[1].plot(contour[:,1], contour[:,0], linewidth=1.5, color='yellow')
            axes[1].set_title(f"Tumor Contour ({label_name})" if label_name else "Tumor Contour")
        else:
            axes[1].imshow(mask_bool, cmap='Reds', alpha=alpha)
            axes[1].set_title(f"Tumor Mask Overlay ({label_name})" if label_name else "Tumor Mask Overlay")
    else:
        axes[1].set_title("No mask provided")

    axes[1].axis('off')
    if pid:
        fig.suptitle(f"Patient ID: {pid}", fontsize=14)
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, bbox_inches='tight', dpi=150)
    plt.show()
    return fig, axes

def show_gradcam_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, cmap: str = 'jet', absolute: bool = False, figsize: Tuple[int,int] = (6,6), title: Optional[str] = None, save: Optional[str] = None):
    if heatmap is None:
        raise ValueError("heatmap is None")

    base_u8 = _to_uint8_image(image)

    hm = np.array(heatmap, dtype=np.float32)
    if absolute:
        hm = np.abs(hm)
    p1, p99 = np.percentile(hm.ravel(), (2.0, 98.0))
    if p99 - p1 <= 0:
        hm_norm = np.zeros_like(hm)
    else:
        hm_norm = (hm - p1) / (p99 - p1)
        hm_norm = np.clip(hm_norm, 0.0, 1.0)

    cmap_obj = plt.get_cmap(cmap)
    heatmap_rgb = (cmap_obj(hm_norm)[:, :, :3] * 255).astype(np.uint8)

    if base_u8.ndim == 2:
        base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_u8[..., :3]

    heatmap_rgb = cv2.resize(heatmap_rgb, (base_rgb.shape[1], base_rgb.shape[0]))

    overlay = cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(overlay)
    ax.set_title(title or "Grad-CAM Overlay")
    ax.axis('off')

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, bbox_inches='tight', dpi=150)
    plt.close(fig)

    return fig, ax


def _mask_to_contour(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if len(contours) > 0:
        cv2.drawContours(canvas, contours, -1, (255, 0, 0), 2)  # red
    return canvas


def plot_grid(images: Sequence[np.ndarray], masks: Optional[Sequence[np.ndarray]] = None, preds: Optional[Sequence[str]] = None, labels: Optional[Sequence[str]] = None, ncols: int = 4, figsize_per: Tuple[int,int] = (3,3), save: Optional[str] = None):
    n = len(images)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(_to_uint8_image(img), cmap='gray')
        title = ""
        if labels is not None:
            title += f"T:{labels[i]}"
        if preds is not None:
            title += (", " if title else "") + f"P:{preds[i]}"
        if title:
            ax.set_title(title, fontsize=8)
        ax.axis('off')
        if masks is not None:
            m = _prepare_mask(masks[i])
            if m is not None:
                # ax.imshow(np.ma.masked_where(~m, m), cmap='Reds', alpha=0.25)
                # sharp boundary instead of filled mask
                contour = _mask_to_contour(m)
                ax.imshow(contour, cmap='Reds', alpha=0.9)

    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, bbox_inches='tight', dpi=150)
    plt.show()
    return fig, axes

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize: bool = False, figsize: Tuple[int,int] = (5,4), save: Optional[str] = None):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, bbox_inches='tight', dpi=150)
    plt.show()
    return fig, ax


def save_prediction_visual(original, gt_mask, pred_mask, save_path, figsize=(12,4), cmap='gray', alpha=0.45):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        img_u8 = _to_uint8_image(original)
    except Exception:
        # fallback: scale manually
        arr = np.asarray(original).astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        img_u8 = (255 * arr).astype(np.uint8)
        if img_u8.ndim == 2:
            img_u8 = np.repeat(img_u8[:, :, None], 3, axis=2)

    gt_bool = _prepare_mask(gt_mask)
    pred_bool = _prepare_mask(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(img_u8)
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(img_u8)
    if gt_bool is not None:
        if _HAS_SKIMAGE:
            contours = measure.find_contours(gt_bool.astype(np.uint8), 0.5)
            for contour in contours:
                axes[1].plot(contour[:,1], contour[:,0], linewidth=1.2, color='yellow')
        else:
            axes[1].imshow(np.ma.masked_where(~gt_bool, gt_bool), cmap='Reds', alpha=alpha)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(img_u8)
    if pred_bool is not None:
        if _HAS_SKIMAGE:
            contours = measure.find_contours(pred_bool.astype(np.uint8), 0.5)
            for contour in contours:
                axes[2].plot(contour[:,1], contour[:,0], linewidth=1.2, color='cyan')
        else:
            axes[2].imshow(np.ma.masked_where(~pred_bool, pred_bool), cmap='Blues', alpha=alpha)
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

