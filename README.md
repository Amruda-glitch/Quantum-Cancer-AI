# Quantum-Cancer-AI

Quantum-Enhanced Medical Imaging for Multi-Class Cancer Detection

An AI-powered medical image analysis pipeline that detects and classifies **9 cancer types**
from medical scan images, combining deep-learning feature extraction with a
**Quantum-Inspired Genetic Algorithm (QIGA)** — built to the interface of the QuantumNow
(BQPhy SDK) — for feature selection and hyperparameter optimization.

---

## 1. Overview

Manual review of CT, MRI, histopathology, and cytology slides is slow, subjective, and
constrained by the number of trained radiologists/pathologists available, especially in
low-resource settings. This project builds a reproducible, end-to-end pipeline that:

- Cleans and validates a large, messy, multi-source cancer image dataset
- Extracts deep CNN features (ResNet50 + EfficientNet-B0)
- Trains and benchmarks classical ML models (Random Forest, SVM, XGBoost) and deep models
  (ResNet50, EfficientNet-B0, Hybrid Ensemble)
- Uses a quantum-inspired optimizer to select the most informative feature subset and tune
  hyperparameters, benchmarked against classical Grid Search and Random Search
- Explains predictions visually using Grad-CAM
- Auto-generates a full technical report and comparative performance table

The 9 supported cancer classes: **Leukemia, Brain Cancer, Breast Cancer, Cervical Cancer,
Kidney Cancer, Colon Cancer, Lung Cancer, Lymphoma, Oral Cancer.**

---

## 2. Key Features

1. **Dataset Validation & Cleaning** — MD5 hashing + OpenCV Laplacian-variance blur detection to flag corrupted, duplicate, and low-quality images.
2. **Mixed-Class Handling (Leukemia)** — separates the `ALL` dataset into Original, Preprocessed, Processed, Augmented, and Suspicious categories.
3. **Medical-Grade Preprocessing** — denoising, CLAHE contrast enhancement, normalization, and ImageNet-standardized scaling.
4. **Data Augmentation** — rotation, flips, zoom, random crop, and brightness adjustments.
5. **Deep Feature Extraction** — 2048-D ResNet50 + 1280-D EfficientNet-B0 fused into a 3328-D feature vector.
6. **Classical Baselines** — Random Forest, SVM, and XGBoost.
7. **Deep Learning Models** — ResNet50, EfficientNet-B0, and Hybrid Ensemble.
8. **Quantum-Inspired Optimization** — QIGA for feature selection and hyperparameter tuning.
9. **Explainable AI (Grad-CAM)** — activation heatmaps overlaid on medical images to highlight the regions influencing the model's prediction.
10. **Automated Evaluation** — confusion matrices, ROC curves, precision, recall, F1-score, and classification reports.
11. **Research Report Generation** — automatically produces publication-ready tables and performance summaries.

EOF
