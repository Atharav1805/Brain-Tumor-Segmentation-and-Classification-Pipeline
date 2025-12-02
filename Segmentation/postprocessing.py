import numpy as np
from skimage import morphology, measure
import cv2


def keep_largest_component(mask):
    mask = mask.astype(np.uint8)
    labels = measure.label(mask, connectivity=2)
    
    if labels.max() == 0:
        return mask

    largest_cc = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1)
    return largest_cc.astype(np.uint8)


def remove_small_regions(mask, min_size=200):
    cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    return cleaned.astype(np.uint8)


def morphological_smoothing(mask, radius=3):
    selem = morphology.disk(radius)
    closed = morphology.closing(mask, selem)
    return closed.astype(np.uint8)


def postprocess_mask(mask, threshold=0.5, min_size=200):
    # 1) Threshold
    bin_mask = (mask >= threshold).astype(np.uint8)

    # 2) Morphological closing (remove small holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    # 3) Keep largest connected component only
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # ignore background
        cleaned = (labels == largest).astype(np.uint8)
    else:
        cleaned = closed

    return cleaned