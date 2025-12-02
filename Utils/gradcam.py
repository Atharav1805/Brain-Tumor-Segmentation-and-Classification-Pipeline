import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2
import os

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return layer.name
        except:
            pass
    raise ValueError("No 4D Conv layer found for Grad-CAM.")


def compute_gradcam(model, img, last_conv_name, class_index=None):
    if img.ndim == 2:
        img_input = np.expand_dims(img, axis=-1)    
    elif img.shape[-1] == 1:
        img_input = img                           
    else:
        raise ValueError(f"Invalid input shape for model: {img.shape}")

    inp = np.expand_dims(img_input, 0).astype("float32")  

    # Grad-CAM model
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)          # (1,Hc,Wc,C)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))   # (C,)
    conv_out = conv_out[0]                         # (Hc,Wc,C)

    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) > 0:
        heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap

def overlay_gradcam(img, heatmap, alpha=0.22, cmap="magma"):
    if img.ndim == 2:
        base = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_GRAY2RGB)
    else:
        base = cv2.cvtColor((img[...,0] * 255).astype("uint8"), cv2.COLOR_GRAY2RGB)

    H, W = base.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    cm = plt.get_cmap(cmap)
    heatmap_rgb = (cm(heatmap_resized)[..., :3] * 255).astype("uint8")

    overlay = cv2.addWeighted(base, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay

def save_gradcam_figure(img, heatmap, title, save_path, cmap="viridis", alpha=0.4):

    overlay = overlay_gradcam(img, heatmap, alpha=alpha, cmap=cmap)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")

    # Add colorbar for heatmap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import Normalize

    cax = inset_axes(ax, width="60%", height="4%", loc="lower center")
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Activation", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
