import os
import sys
import json
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
import cv2

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score


if "COLAB_GPU" in os.environ:
    BASE_DIR = "/content/"
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "Data/converted_npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs/Classifier_Cropped")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "folds"), exist_ok=True)

# Config
IMG_SIZE = 256
EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_FOLDS = 5
PAD = 32
SEED = 42
CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.experimental.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)


sys.path.append(BASE_DIR)

from Utils.dataset_loader import load_figshare_npy_dataset
from Models.custom_cnn import build_custom_cnn   



def safe_bbox(mask, pad=16):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:  
        # fallback: center crop
        h, w = mask.shape
        cx, cy = w//2, h//2
        r = min(h, w)//4
        return cx-r, cy-r, cx+r, cy+r

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    x0 -= pad; y0 -= pad
    x1 += pad; y1 += pad

    x0 = max(0, x0)
    y0 = max(0, y0)
    return x0, y0, x1, y1


def crop_tumor(img, mask, pad=16, out_size=256):
    if img.ndim == 3:
        img = img[:, :, 0]  # grayscale

    x0, y0, x1, y1 = safe_bbox(mask, pad)
    crop = img[y0:y1, x0:x1]

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    if crop.max() > 1:
        crop = crop / 255.0

    return np.expand_dims(crop.astype(np.float32), axis=-1)



def prepare_dataset():
    print("🔹 Loading dataset (preprocess=True)...")
    X, y, masks, pids = load_figshare_npy_dataset(DATA_DIR, preprocess=True)

    print(f"✔ X={X.shape}, y={np.unique(y)}")

    print("🔹 Cropping tumors...")
    X_crop = np.zeros((len(X), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    for i in range(len(X)):
        X_crop[i] = crop_tumor(X[i], masks[i], pad=PAD, out_size=IMG_SIZE)

    print("Cropped dataset shape:", X_crop.shape)
    return X_crop, y


def build_model():
    model = build_custom_cnn(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        num_classes=3
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_fold(fold, train_idx, val_idx, X, y):
    print(f"\n{fold+1}/{NUM_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model()

    fold_dir = os.path.join(OUTPUT_DIR, "folds", f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(fold_dir, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt, early],
        verbose=1
    )

    np.save(os.path.join(fold_dir, "history.npy"), history.history)

    preds = np.argmax(model.predict(val_ds, verbose=0), axis=1)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")

    json.dump(
        {"accuracy": acc, "f1_macro": f1},
        open(os.path.join(fold_dir, "metrics.json"), "w"),
        indent=2
    )

    print(f"Fold {fold+1} — ACC: {acc:.4f}  F1: {f1:.4f}")
    return acc, f1


def main():
    X, y = prepare_dataset()

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    accs, f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        acc, f1 = train_fold(fold, train_idx, val_idx, X, y)
        accs.append(acc)
        f1s.append(f1)

    summary = {
        "avg_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "avg_f1_macro": float(np.mean(f1s)),
        "std_f1_macro": float(np.std(f1s)),
        "timestamp": datetime.now().isoformat()
    }

    json.dump(
        summary,
        open(os.path.join(OUTPUT_DIR, "cv_summary.json"), "w"),
        indent=2
    )

    print("\nFINAL RESULTS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
