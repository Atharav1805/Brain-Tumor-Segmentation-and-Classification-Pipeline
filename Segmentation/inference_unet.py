import os
import numpy as np
import tensorflow as tf
import cv2

from Utils.dataset_loader import load_figshare_npy_dataset
from Segmentation.postprocessing import postprocess_mask
from Models.unet_model import build_unet
from Utils.visualization import save_prediction_visual

IMG_SIZE = 256
PREPROCESS = True  
SUFFIX = "preprocessed" if PREPROCESS else "raw"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data", "converted_npy")
MODEL_PATH = os.path.join(BASE_DIR, "Outputs", f"UNet_{SUFFIX}", "best_unet.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs", f"UNet_{SUFFIX}", "inference_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("🔹 Loading dataset...")
    X, y, masks, pids = load_figshare_npy_dataset(DATA_DIR, preprocess=PREPROCESS)

    X_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X], dtype=np.float32)
    X_gray = X_gray / 255.0
    X_gray = np.expand_dims(X_gray, -1)

    print("🔹 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Running inference...")
    preds = model.predict(X_gray, batch_size=4)
    preds = preds[..., 0] 

    for idx in range(len(preds)):
        raw_pred = preds[idx]
        post = postprocess_mask(raw_pred)

        img = (X_gray[idx][:,:,0] * 255).astype("uint8")
        gt = masks[idx][:,:,0] if masks is not None else None

        out_path = os.path.join(OUTPUT_DIR, f"pred_{idx}.png")
        save_prediction_visual(img, gt, post*255, out_path)

    print("Inference complete. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True, help="Path to trained U-Net .h5 file")
#     parser.add_argument("--image", type=str, required=True, help="Path to .npy image file")
#     parser.add_argument("--save", type=str, default=None, help="Path to save visualization PNG")

#     args = parser.parse_args()

#     model = load_unet_model(args.model)
#     predict_single_image(model, args.image, args.save)
