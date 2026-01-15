# Explainable Lung Cancer Detection Using CNNs

This project presents a deep learning–based approach for detecting lung cancer from CT scan images using Convolutional Neural Networks (CNNs). The model is designed as a **binary classifier (Cancer vs Normal)** to align with real-world clinical screening needs and emphasizes **high recall** to minimize missed cancer cases.

---

## Objective
- Detect lung cancer from CT scan images
- Prioritize **sensitivity (recall)** over raw accuracy
- Provide **visual explanations** using Grad-CAM to improve trust and interpretability

---

## Dataset
- Source: Kaggle – *CT Scan Images for Lung Cancer*
- Original classes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, benign, malignant, normal
- Reformulated into:
  - **Cancer (1)**: all cancer-related classes
  - **Normal (0)**

> Dataset is not included due to licensing restrictions.

---

## Model Architecture
- Input: 224×224×3 CT images
- 3 Convolutional layers (32, 64, 128 filters)
- ReLU + MaxPooling
- Fully connected layers with Dropout
- Sigmoid output for binary classification

---

## Training Details
- Optimizer: Adam  
- Loss: Binary Cross-Entropy  
- Batch size: 32  
- Epochs: 25  
- Decision Threshold: **0.3** (to improve recall)

---

## Performance
| Metric | Value |
|------|------|
| Accuracy | 90% |
| Recall (Sensitivity) | 91.5% |
| Specificity | 87% |
| ROC-AUC | 0.949 |
| PR-AUC | 0.982 |
| MCC | 0.759 |

---

## Explainability (Grad-CAM)
Grad-CAM visualizations confirm that the model focuses on meaningful lung regions:
- **Cancer CT scans** → localized high activation
- **Normal CT scans** → diffuse activation

<p align="center">
  <img src="figures/gradcam_cancer.png" width="45%">
  <img src="figures/gradcam_normal.png" width="45%">
</p>

---

## ROC & PR Curves
<p align="center">
  <img src="figures/roc_curve.png" width="45%">
  <img src="figures/pr_curve.png" width="45%">
</p>

---

## Paper
The complete methodology and results are documented in IEEE format:

`paper/lung_cancer_detection_ieee.pdf`

---

## Key Takeaways
- Binary reformulation improves clinical relevance
- Threshold tuning enhances screening safety
- Explainability increases trust for medical AI

---

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- Scikit-learn

---

## Author
**Aishwarya Patil**  
Computer Science & Engineering  
Manipal Institute of Technology, Bengaluru
