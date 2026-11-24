# Intelligent Microscopic Blood Cell Classification for Leukemia Detection

## Overview
This project focuses on classifying microscopic blood cell images into five categories using deep learning and machine learning techniques. It evaluates CNN-ML, CBAM-DenseNet19, and Vision Transformer architectures and deploys the classification system through a Flask web application named CellVision.

## Dataset
- **Dataset:** Blood Cell Images for Cancer Detection – Kaggle
- **Total Images:** 5,000
- **Classes:** Basophil, Erythroblast, Monocyte, Myeloblast, and Segmented Neutrophil.
- **Image Resolution:** 224 × 224 × 3

## Project Workflow
1. **Dataset Analysis:** Perform statistical analysis and EDA to examine class distribution, image properties, pixel intensities, and morphological characteristics.
2. **Preprocessing:** Apply image resizing, sharpening, and normalization.
3. **Dataset Splitting:** Use stratified splitting with 70% training, 15% validation, and 15% testing.
4. **Model Development:** Implement and evaluate:
   - CNN-ML (CNN Feature Extractor + Random Forest)
   - CBAM-DenseNet19
   - Vision Transformer (ViT)
5. **Model Evaluation:** Analyse Accuracy, Precision, Recall, F1-Score, and Log Loss using learning curves, classification reports, confusion matrices, and precision-recall curves.
6. **Deployment:** Develop CellVision, a Flask-based application for microscopic blood cell classification.

## Model Performance

| Model | Accuracy | F1-Score |
|---|---:|---:|
| CNN-ML | 96.93% | 96.93% |
| CBAM-DenseNet19 | 74.67% | 68.30% |
| Vision Transformer | 87.47% | 87.49% |

## Technologies Used
Python, TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Random Forest, Joblib, and Flask.

## Project Structure
```text
Blood-Cell-Classification/
│
├── Code.ipynb
├── Results/
│
└── WebApp/
    ├── app.py
    ├── models/
    │   ├── cnn_ml_best.keras
    │   └── cnn_ml_random_forest.joblib
    ├── requirements.txt
    ├── templates/
    │   └── index.html
    └── static/
        ├── css/
        ├── js/
        └── uploads/

## Objective
To develop an automated microscopic blood cell classification system using deep learning and machine learning approaches, supporting research in computer-assisted leukemia screening.
