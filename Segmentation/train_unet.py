import os, sys, random, json
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime

import os
import sys
import numpy as np
import tensorflow as tf

# Config and setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_ROOT = BASE_DIR

PREPROCESS = True
SUFFIX = "preprocessed" if PREPROCESS else "raw"
DATA_DIR = "/content/CS663_Project/Data/converted_npy"   # change if needed
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"UNet_{SUFFIX}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.append(BASE_DIR)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print(f"GPUs detected: {len(gpus)}")

IMG_SIZE = 256
EPOCHS = 40
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
SEED = 42
SAMPLES_TO_SAVE = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


from Utils.dataset_loader import load_figshare_npy_dataset
from Utils.visualization import save_prediction_visual

from Models.unet_model import build_unet
from postprocessing import postprocess_mask
from Utils.metrics import dice_coef, iou_coef

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def main():
    print("🔹 Loading dataset...")
    X, y, masks, pids = load_figshare_npy_dataset(DATA_DIR, preprocess=PREPROCESS)

    print("Converting preprocessed RGB → GRAYSCALE...")
    X_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X], dtype=np.float32)
    X_gray = X_gray / 255.0
    X_gray = np.expand_dims(X_gray, -1)

    masks = np.array(masks, dtype=np.uint8)
    masks = np.expand_dims(masks, -1)
    masks = masks.astype("float32")

    idxs = np.arange(len(X_gray))
    np.random.shuffle(idxs)

    split = int(0.8 * len(idxs))
    train_idx, val_idx = idxs[:split], idxs[split:]

    x_train, y_train = X_gray[train_idx], masks[train_idx]
    x_val, y_val = X_gray[val_idx], masks[val_idx]

    print(f"Train={len(x_train)} | Val={len(x_val)} | Input={x_train.shape}")

    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coef, iou_coef]
    )

    ckpt_path = os.path.join(OUTPUT_DIR, "best_unet.keras")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_dice_coef", save_best_only=True, mode="max", verbose=1
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef", patience=10, mode="max", restore_best_weights=True
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=[ckpt, early], verbose=1
    )

    np.save(os.path.join(OUTPUT_DIR, "history.npy"), history.history)
    sample_idxs = np.random.choice(len(x_val), SAMPLES_TO_SAVE, replace=False)

    for idx in sample_idxs:

        img_orig = (x_val[idx][:, :, 0] * 255).astype("uint8")
        gt = y_val[idx][:, :, 0].astype("uint8")
        raw_pred = model.predict(np.expand_dims(x_val[idx], 0))[0, :, :, 0]
        pred_mask = postprocess_mask(raw_pred).astype("uint8")

        out_path = os.path.join(OUTPUT_DIR, "samples", f"sample_{idx}.png")
        save_prediction_visual(img_orig, gt, pred_mask, out_path)


    print("Segmentation training complete.")

if __name__ == "__main__":
    main()
