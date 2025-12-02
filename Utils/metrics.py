import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def classification_metrics(y_true, y_pred, average='macro'):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(title)
    plt.show()
    return cm

def gradcam_iou(gradcam_map, mask, threshold=0.5):
    gradcam_map = gradcam_map / (gradcam_map.max() + 1e-8)
    gradcam_bin = (gradcam_map >= threshold).astype(np.uint8)
    mask_bin = (mask > 0).astype(np.uint8)

    intersection = np.logical_and(gradcam_bin, mask_bin).sum()
    union = np.logical_or(gradcam_bin, mask_bin).sum()
    iou = intersection / (union + 1e-8)

    return round(iou, 4)


import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = (
        tf.reduce_sum(y_true_f)
        + tf.reduce_sum(y_pred_f)
        - intersection
        + smooth
    )
    return (intersection + smooth) / union
