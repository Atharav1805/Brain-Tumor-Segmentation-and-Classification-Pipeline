import os
import random
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.image_preprocessing import preprocess_mri, visualize_preprocessing


DATA_DIR = "/content/CS663_Project/Data/converted_npy"   # change if needed
NUM_SAMPLES = 5               
RESIZE_SHAPE = (224, 224)
DENOISE_METHOD = "anisotropic"   
APPLY_CLAHE = True
SAVE_RESULTS = True              

image_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_image.npy")])
print(f"Found {len(image_files)} MRI slices in {DATA_DIR}")

if not image_files:
    raise FileNotFoundError("No *_image.npy files found. Check DATA_DIR path.")

samples = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

os.makedirs("Outputs/preprocessing_tests", exist_ok=True)

for i, fname in enumerate(samples, 1):
    base = fname.replace("_image.npy", "")
    img_path = os.path.join(DATA_DIR, fname)
    mask_path = os.path.join(DATA_DIR, f"{base}_mask.npy") if os.path.exists(os.path.join(DATA_DIR, f"{base}_mask.npy")) else None

    print(f"\nProcessing sample {i}/{len(samples)}: {fname}")

    img = np.load(img_path).astype(np.float32)
    mask = np.load(mask_path) if mask_path else None

    preprocessed_img = preprocess_mri(
        img,
        resize_shape=RESIZE_SHAPE,
        denoise_method=DENOISE_METHOD,
        apply_clahe=APPLY_CLAHE,
        debug=True
    )

    vis = (preprocessed_img - preprocessed_img.min()) / (preprocessed_img.max() - preprocessed_img.min() + 1e-8)
    visualize_preprocessing(img, vis, title_suffix=f"({DENOISE_METHOD}{'+CLAHE' if APPLY_CLAHE else ''})")

    if SAVE_RESULTS:
        save_path = f"Outputs/preprocessing_tests/{base}_compare.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Original"); axes[0].axis('off')
        axes[1].imshow(vis)
        axes[1].set_title(f"Preprocessed ({DENOISE_METHOD})"); axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=250)
        plt.close(fig)
        print(f"💾 Saved comparison to {save_path}")
