import os
import numpy as np
import tensorflow as tf
import random
import cv2

from Utils.dataset_loader import load_figshare_npy_dataset
from Utils.gradcam import compute_gradcam, get_last_conv_layer
from Utils.visualization import save_gradcam_figure

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]
IMG_SIZE = 256
PREPROCESS = True
SUFFIX = "preprocessed" if PREPROCESS else "raw"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data", "converted_npy")

MODEL_PATH = os.path.join(BASE_DIR, "Outputs", f"Classifier_{SUFFIX}", "folds", "fold_1", "best_classifier.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs", f"Classifier_{SUFFIX}", "inference_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("🔹 Loading dataset...")
    X, y, masks, pids = load_figshare_npy_dataset(DATA_DIR, preprocess=PREPROCESS)

    print("🔹 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    last_conv = get_last_conv_layer(model)

    print("Running inference...")
    probs = model.predict(X, batch_size=8)
    preds = np.argmax(probs, axis=1)

    np.save(os.path.join(OUTPUT_DIR, "preds.npy"), preds)
    np.save(os.path.join(OUTPUT_DIR, "probs.npy"), probs)

    print("🔹 Generating Grad-CAM...")
    gradcam_dir = os.path.join(OUTPUT_DIR, "gradcam_examples")
    os.makedirs(gradcam_dir, exist_ok=True)

    chosen = random.sample(range(len(X)), min(12, len(X)))

    for idx in chosen:
        img = X[idx]
        heatmap = compute_gradcam(model, img, last_conv, class_index=preds[idx])
        mask = masks[idx] if masks is not None else None

        out_path = os.path.join(gradcam_dir, f"sample_{idx}_pred{preds[idx]}.png")
        save_gradcam_figure(img, heatmap, CLASS_NAMES[preds[idx]], out_path, mask=mask)

    print("Classification inference complete. Saved results to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
