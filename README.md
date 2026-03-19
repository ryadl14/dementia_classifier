# Dementia Classification from MRI using CNNs

A deep learning pipeline that classifies dementia severity from 2D brain MRI scans. Three CNN architectures are evaluated iteratively — each built to address the shortcomings of the last — culminating in a fine-tuned ResNet50 model achieving **81% accuracy** and a **macro-average F1-score of 0.78**.

---

## Overview

Dementia affects ~850,000 people in the UK and is estimated to triple by 2050. Manual evaluation of MRI volumes is time-consuming and expensive. This project explores whether convolutional neural networks can reliably classify dementia severity from 2D MRI slices, and critically examines the pitfalls of standard evaluation metrics on imbalanced medical datasets.

**Classes:** `MildImpairment` · `ModerateImpairment` · `NoImpairment`  
**Dataset:** 5,076 MRI images — inherently imbalanced (50.4% No, 35.3% Mild, 14.3% Moderate)

---

## Models

| Model | Architecture | Accuracy | Macro F1 | AUC |
|-------|-------------|----------|----------|-----|
| Model 1 | Baseline CNN (1 conv layer) | 71% | 0.65 | 0.869 |
| Model 2 | Augmented CNN (3 conv layers + class weights) | 50% | 0.41 | 0.703 |
| Model 3 | Fine-tuned ResNet50 (transfer learning) | **81%** | **0.78** | **0.936** |

### Key findings

- **Model 1** revealed a ROC-AUC paradox: Moderate Impairment achieved the highest AUC (0.87) despite a recall of only 0.50. On imbalanced datasets, AUC alone is misleading — per-class analysis is essential.
- **Model 2** demonstrated that aggressive data augmentation (random zoom/rotation) destroys the subtle spatial markers of cortical atrophy needed to distinguish closely related classes, causing performance to collapse below random guessing for Mild Impairment (AUC: 0.36).
- **Model 3** used ImageNet-pretrained ResNet50 with the final 5 layers unfrozen for domain adaptation. Transfer learning successfully mapped biological markers of brain atrophy, resolving the minority-class failures of the earlier models.

---

## Repository Structure

```
dementia-mri-classifier/
│
├── notebooks/
│   └── AIML_Assignment_Notebook.ipynb   # Full pipeline: data loading → 3 models → evaluation
│
├── data/
│   └── raw/                             # Place dataset here (see setup below)
│       ├── MildImpairment/
│       ├── ModerateImpairment/
│       └── NoImpairment/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/dementia-mri-classifier.git
cd dementia-mri-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
The dataset is not included in this repository. Download the **Alzheimer MRI Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

Extract the contents into `data/raw/` so the directory contains three subfolders: `MildImpairment/`, `ModerateImpairment/`, and `NoImpairment/`.

> **Running on Google Colab?** Mount your Google Drive and update the `zip_path` and `extract_path` variables in the data loading cell to point to your Drive location.

### 4. Launch the notebook
```bash
jupyter notebook notebooks/AIML_Assignment_Notebook.ipynb
```

---

## Results

### Model 3 — Confusion Matrix
Model 3 achieved the strongest per-class performance, correctly classifying 152/180 Mild, 44/72 Moderate, and 209/251 No Impairment cases.

### Comparative ROC Curves
Micro-average AUC scores: Model 3 (0.936) vs Model 1 (0.869) vs Model 2 (0.703).

---

## Limitations & Future Work

- **2D slices** discard volumetric context. Future iterations should use patient-stratified 3D MRI volumes with 3D-CNN architectures (Chen et al., 2025 achieved 92.33% with 3D-ResNet).
- **Class imbalance** was partially addressed via class weights. Targeted minority oversampling (SMOTE or augmentation-only on underrepresented classes) could push performance beyond 90%.
- **Explainability** — Grad-CAM heatmaps should be integrated to spatially verify that predictions are driven by biologically plausible regions (e.g. ventricular enlargement, cortical thinning), not artefacts.
- **Ethical considerations** — Black-box clinical tools carry real liability and bias risks. Any deployment in an NHS context must be paired with Explainable AI, diverse representative training data, and a clear regulatory framework.

---

## Dependencies

See `requirements.txt`. Core stack: TensorFlow 2.x, scikit-learn, seaborn, matplotlib.

---

## Academic Context

Developed as coursework for MSc Applied Bioinformatics at King's College London (Jan–Mar 2026).
