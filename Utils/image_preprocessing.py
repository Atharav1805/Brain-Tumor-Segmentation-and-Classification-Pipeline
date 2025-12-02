import numpy as np
import cv2
import matplotlib.pyplot as plt
from medpy.filter import smoothing


def normalize_image(img, mode = 'min-max'):
    img = img.astype(np.float32)
    if mode=='z-score':
      img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    elif mode == 'min-max':
      img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    else:
      print("Invalid mode chosen")
      return
    
    return img


def anisotropic_diffusion_denoise(img, niter=6, kappa=40, gamma=0.08):
    img = img.astype(np.float32)
    denoised = smoothing.anisotropic_diffusion(img, niter=niter, kappa=kappa, gamma=gamma)
    return np.clip(denoised, 0, 1)


def bilateral_filter_denoise(img, d=9, sigma_color=75, sigma_space=75):
    img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    img_uint8 = np.uint8(img_norm * 255)
    denoised = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
    return denoised.astype(np.float32) / 255.0


def apply_CLAHE(img, clip_limit=1.5, tile_grid_size=(8,8)):
    img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    img_uint8 = np.uint8(img_norm * 255)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_uint8)
    return enhanced.astype(np.float32) / 255.0


def preprocess_mri(img, resize_shape=(256, 256), denoise_method='anisotropic', apply_clahe=False, debug=False):
    img = img.astype(np.float32)
    img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
    img = normalize_image(img, mode = 'min-max')

    if denoise_method == 'anisotropic':
        img = anisotropic_diffusion_denoise(img, niter=5, kappa=25, gamma=0.05)
    elif denoise_method == 'bilateral':
        img = bilateral_filter_denoise(img)
    elif denoise_method == 'none':
        pass
    else:
        raise ValueError("denoise_method must be 'anisotropic', 'bilateral', or 'none'")

    if apply_clahe:
        img = apply_CLAHE(img, clip_limit=1.2, tile_grid_size=(14,14))

    img = normalize_image(img, mode = 'z-score')
    # img_rgb = np.stack([img] * 3, axis=-1)

    if debug:
        print(f"Preprocessed image: shape={img.shape}, range=({img.min():.3f}, {img.max():.3f})")
    return img


def visualize_preprocessing(original_img, preprocessed_img, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original MRI"); axes[0].axis('off')

    axes[1].imshow(preprocessed_img)
    axes[1].set_title(f"Preprocessed MRI {title_suffix}"); axes[1].axis('off')

    axes[2].hist(preprocessed_img.ravel(), bins=50, color='gray')
    axes[2].set_title("Pixel Intensity Histogram")
    plt.tight_layout()
    plt.show()
