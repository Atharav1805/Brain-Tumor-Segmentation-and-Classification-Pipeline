# 🧠 Brain Tumor Segmentation & Classification Pipeline

### Deep Learning · MRI Analysis · TensorFlow/Keras

This repository contains a complete, end-to-end deep learning pipeline for brain tumor segmentation and classification using axial MRI scans.

It includes training scripts, inference modules, preprocessing utilities, visualization tools, model architectures, evaluation methods, and full pipeline outputs.

A cleaned and standardized NumPy MRI dataset is also provided on Kaggle:
➡️ **Dataset (Kaggle):** [Cleaned Brain Tumor MRI Dataset (NumPy Version)](https://www.kaggle.com/datasets/atharavsonawane/cleaned-brain-tumor-mri-dataset-numpy-version)

---

## 🌟 Features

* **UNet-based tumor segmentation**
* **Custom CNN & EfficientNet classifier**
* Exact training-consistent preprocessing (z-score + scaling)
* Automatic ROI extraction using segmentation mask
* FP/FN visualization + contour overlays
* Full A4 scientific 6-panel pipeline figure
* GRAD-CAM visualization for classifier interpretability
* Training/evaluation scripts for both models
* Clean directory structure for reproducibility
* Performance curves (Dice, IoU, loss, accuracy)
* Utility scripts for noise analysis, metrics, plotting

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
