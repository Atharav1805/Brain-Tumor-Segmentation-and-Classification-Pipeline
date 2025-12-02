import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from Utils.image_preprocessing import preprocess_mri, normalize_image

def load_figshare_npy_dataset(data_dir: str, preprocess: bool = True, denoise_method: str = "anisotropic", apply_clahe: bool = True, resize_shape=(256, 256), global_norm: bool = False):
   
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_image.npy")])
    X_data, y_data, masks, pids = [], [], [], []

    print(f"Loading dataset from {data_dir} ({len(image_files)} images)...")

    for img_file in image_files:
        base = img_file.replace("_image.npy", "")
        try:
            img = np.load(os.path.join(data_dir, f"{base}_image.npy"))
            label_arr = np.load(os.path.join(data_dir, f"{base}_label.npy"))
            label = int(np.round(label_arr.flatten()[0])) - 1  # ensure 0-based labels
            mask = np.load(os.path.join(data_dir, f"{base}_mask.npy"))
            pid_arr = np.load(os.path.join(data_dir, f"{base}_PID.npy"))

            try:
                if pid_arr.dtype.kind in {'U', 'S'}:
                    pid = str(pid_arr.item())
                else:
                    chars = []
                    for x in pid_arr.flatten():
                        try:
                            val = int(x)
                            if 32 <= val <= 126:
                                chars.append(chr(val))
                        except Exception:
                            pass
                    pid = ''.join(chars).strip() or str(pid_arr.flatten()[0])
            except Exception:
                pid = "unknown"


            if preprocess:
                img = preprocess_mri(
                    img,
                    resize_shape=resize_shape,
                    denoise_method=denoise_method,
                    apply_clahe=apply_clahe
                )
            else:
                img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
                # img = normalize_image(img)
                # img = np.stack([img]*3, axis=-1)
                
            mask = cv2.resize(mask.astype(np.uint8), resize_shape, interpolation=cv2.INTER_NEAREST)

            X_data.append(img)
            y_data.append(label)
            masks.append(mask)
            pids.append(pid)

        except Exception as e:
            print(f"Skipping {base}: {e}")
            continue

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    if global_norm:
        mean, std = X.mean(), X.std()
        X = (X - mean) / (std + 1e-8)

    assert not np.isnan(X).any(), "Dataset contains NaN values!"
    assert y.min() >= 0 and y.max() < 3, f"Invalid class labels: {np.unique(y)}"
    print(f"Loaded {len(X)} samples successfully.")
    return X, y, masks, pids


def create_train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state)
    rel_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - rel_val_size, stratify=y_temp, random_state=random_state)

    print(f"Split summary: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_kfold_splits(X, y, n_splits=5, random_state=42, shuffle=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    print(f"Performing {n_splits}-fold stratified cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"   Fold {fold+1}: Train={len(train_idx)} | Val={len(val_idx)}")
        yield fold, train_idx, val_idx


def group_by_patient(pids):
    pid_to_indices = {}
    for idx, pid in enumerate(pids):
        pid_to_indices.setdefault(pid, []).append(idx)
    return pid_to_indices
