# 🧠 Brain Tumor Segmentation & Classification Pipeline

### Deep Learning · MRI Analysis · TensorFlow/Keras

This repository contains a complete, end-to-end deep learning pipeline for brain tumor segmentation and classification using axial MRI scans.

A cleaned NumPy MRI dataset is also available:  
➡️ **Kaggle Dataset:** https://www.kaggle.com/datasets/atharavsonawane/cleaned-brain-tumor-mri-dataset-numpy-version

---

## 🌟 Features

- UNet-based tumor segmentation  
- Custom CNN & EfficientNet classifier  
- Training-consistent preprocessing (z-score + scaling)  
- Automatic ROI extraction  
- Full A4 multi-panel visualization of the pipeline  
- Grad-CAM interpretability  
- Cross-validation metrics + performance curves  
- High-quality overlays, prob-maps, and samples  

---

# 📌 Pipeline Overview

## 🔷 **Workflow Summary**

<p align="center">
  <img src="unet_flowchart.png" width="700">
</p>

---

# 1️⃣ Segmentation (UNet)

<p align="center">
  <img src="unet_architecture_horizontal.png" width="700">
</p>

### Steps:
- MRI → z-score normalization  
- Resize + channel formatting  
- UNet produces probability heatmap  
- Post-processing: thresholding, contours, connected components, morphology  

### **Training Curves**
<p align="center">
  <img src="dice_curve.png" width="400">  
  <img src="iou_curve.png" width="400">  
</p>

---

# 2️⃣ ROI Extraction

A bounding box is computed from the segmentation mask, padded, cropped, and preprocessed exactly as used during classifier training.

### Example (from `Full pipeline results/`)
<p align="center">
  <img src="Full pipeline results/130_image_20251123_184416_overlay.png" width="350">
  <img src="Full pipeline results/130_image_20251123_184416_mask.png" width="350">
  <img src="Full pipeline results/130_image_20251123_184416_crop.png" width="350">
</p>

---

# 3️⃣ Classification (Custom CNN / EfficientNet)

<p align="center">
  <img src="custom_cnn_architecture_horizontal.png" width="700">
</p>

### Predicts tumor class:
- **Meningioma**
- **Glioma**
- **Pituitary Tumor**

### Classifier Training Curves
<p align="center">
  <img src="accuracy_curve.png" width="400">
  <img src="loss_curve.png" width="400">
</p>

---

## 🔥 Grad-CAM Visualization

Example Grad-CAMs show attention regions used by the classifier:

<p align="center">
  <img src="Evaluation/Classifier/Fold_1/gradcam/sample_3016_true2_pred2.png" width="350">
  <img src="Evaluation/Classifier/Fold_1/gradcam/sample_1728_true1_pred1.png" width="350">
</p>

---

# 🔻 4️⃣ Full Pipeline Visualization

(`Utils/full_pipeline_viz.py`)

A six-panel scientific figure is generated for every input MRI:

<p align="center">
  <img src="Full pipeline results/sample_130.png" width="800">
</p>

This includes:
- Original Image  
- Predicted Mask  
- Overlay  
- Probability Heatmap  
- Cropped ROI  
- Classifier Output Bar Chart  

---

# 📊 Evaluation Tools

Available in `Evaluation/` and `Utils/`:

- Dice, IoU, pixel accuracy  
- Boxplots & per-sample metrics  
- Confusion matrix  
- ROC / Precision–Recall curves  
- Calibration curve  
- t-SNE visualization  

### Examples:

<p align="center">
  <img src="Evaluation/Classifier/Fold_1/plots/confusion_matrix.png" width="400">
  <img src="Evaluation/Classifier/Fold_1/plots/roc.png" width="400">
</p>

---


## 📂 Repository Structure

```text
.
├── Classification/
│   ├── train_classifier.py
│   └── inference_classifier.py
│
├── Segmentation/
│   ├── train_unet.py
│   ├── inference_unet.py
│   ├── eval_segmentation.py
│   └── postprocessing.py
│
├── Models/
│   ├── unet_model.py
│   ├── custom_cnn.py
│   ├── efficientnet.py
│   └── __init__.py
│
├── Utils/
│   ├── dataset_loader.py
│   ├── image_preprocessing.py
│   ├── visualization.py
│   ├── full_pipeline_viz.py
│   ├── gradcam.py
│   ├── metrics.py
│   ├── plot.py
│   └── test_preprocessing.py
│
├── Data/
│   ├── raw/
│   ├── converted_npy/
│   ├── matlab_to_npy/
│   └── helper scripts
│
├── Evaluation/
│   ├── UNet_preprocessed/
│   └── Classifier/
│
├── Outputs/
│   ├── UNet_preprocessed/
│   ├── preprocessing_tests/
│   └── Classifier/
│
├── Full pipeline results/
│   ├── *_mask.png
│   ├── *_overlay.png
│   ├── *_crop.png
│   ├── *_probmap.png
│   └── *_meta.json
│
├── Flowcharts & Diagrams
│   ├── unet_flowchart.png
│   ├── custom_cnn_flowchart.png
│   ├── unet_architecture_horizontal.png
│   └── custom_cnn_architecture_horizontal.png
│
├── Curves & Metrics
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── dice_curve.png
│   ├── iou_curve.png
│   └── noise_vs_accuracy.png
│
└── Project Documentation.pdf
```

# 🚀 End-to-End Pipeline Summary

## 1️⃣ Segmentation (UNet)
- **MRI → z-score normalization**
- **Resize + channel formatting**
- **UNet model outputs tumor probability map**
- **Post-processing includes:**
  - thresholding
  - contour extraction
  - largest-component selection
  - morphological cleanup

## 2️⃣ ROI Extraction
- Bounding box computed from segmentation mask
- ROI padded and cropped
- Preprocessing applied identically to classifier training

## 3️⃣ Classification (Custom CNN / EfficientNet)
Predicts tumor type:
- **Meningioma**
- **Glioma**
- **Pituitary Tumor**

## 4️⃣ Full Pipeline Visualization (A4 Scientific Figure)
`Utils/full_pipeline_viz.py` generates a six-panel figure:
- Original Image
- Overlay (GT & Prediction)
- Predicted Mask
- Probability Heatmap
- Cropped ROI
- Classifier Output Bar Chart

Each run also saves metadata (`*.json`) and intermediate images.

# 📊 Evaluation Tools
Available in `Utils/` and `Evaluation/`:
- Dice coefficient
- IoU
- Pixel accuracy
- Confusion matrices
- Model performance curves
- Noise–robustness evaluation
- Preprocessing consistency checks
- GRAD-CAM for classifier interpretability

# 📁 Dataset
A cleaned and standardized version of the **Brain Tumor MRI Dataset** is available here:  
➡️ **Kaggle: Cleaned Brain Tumor MRI Dataset (NumPy Version)**

**Features:**
- Converted to `.npy`
- Consistent naming
- Resized & normalized
- Train/val/test ready
- Suitable for both segmentation & classification pipelines

# 🔧 Installation
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn tqdm
```
Optional:

```bash
pip install albumentations
```

# 🖥️ Running Inference

## 🔹 Segmentation
```bash
python Segmentation/inference_unet.py \
    --input image.npy \
    --model Models/unet_model.keras
```
## 🔹 Classification
```bash
python Classification/inference_classifier.py \
    --input roi.npy \
    --model Models/custom_cnn.keras
```
## 🔹 Full Pipeline (Recommended)
```bash
python Utils/full_pipeline_viz.py \
    --input image.npy \
    --seg Models/unet_model.keras \
    --clf Models/custom_cnn.keras \
    --out result.png
```

# 📜 Documentation

A complete project write-up is available in:

Project Documentation.pdf

Includes:

- Literature review

- Architecture details

- Training methodology

- Dataset preparation

- Results & discussion

- Visualizations

# 👤 Author

Atharav Sonawane
Deep Learning · Computer Vision · Medical Imaging

Kaggle: https://www.kaggle.com/atharavsonawane

GitHub: https://github.com/Atharav1805
