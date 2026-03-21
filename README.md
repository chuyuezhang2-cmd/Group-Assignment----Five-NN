# 🚢 Titanic Survival Prediction  
**Machine Learning for Business**  
**Group Name**: Five-NN  

[![Python 3.9.7](https://img.shields.io/badge/Python-3.9.7-blue.svg)](https://www.python.org)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)  
[![Kaggle Titanic](https://img.shields.io/badge/Kaggle-Titanic-20B2AA.svg)](https://www.kaggle.com/c/titanic)

---

## 📌 Project Overview

This project predicts passenger survival in the 1912 Titanic disaster — a classic **binary classification** task (Survived: 0 = No, 1 = Yes).

We built a complete **end-to-end ML pipeline** featuring:
- Thorough Exploratory Data Analysis (EDA)
- Advanced data preprocessing & feature engineering
- 7-model comparison with hyperparameter tuning (GridSearchCV + 5-fold CV)
- Detailed model interpretation, error analysis & final selection

**Final Model**: Logistic Regression  
**5-Fold CV Performance**:
- **Accuracy**: 82.12%
- **F1-score**: 0.754
- **AUC-ROC**: 0.858

These results beat typical Titanic baselines (~78–80%) and clearly show the power of gender, passenger class, and our engineered features (Title, FamilySize, Age×Pclass, etc.).

---

## 📋 Table of Contents
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation-reproducibility)
- [How to Run & Reproduce](#how-to-run--reproduce-results)
- [Loading the Trained Model](#loading-the-trained-model)
- [Generating Kaggle Submission](#generating-kaggle-submission)
- [Full Project Report](#full-project-report)
- [Technologies Used](#technologies-used)
- [Feedback](#feedback)

---

## 📁 Repository Structure

```text
titanic-survival-prediction/
├── data/                          # Raw & preprocessed CSV files
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessed_train.csv
│   └── preprocessed_test.csv
├── notebooks/                     # Step-by-step Jupyter workflows
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing.ipynb
│   ├── 3_Modeling_Comparison.ipynb
│   └── 4_Final_Model_Evaluation.ipynb
├── figures/                       # Visualizations used in report
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── src/                           # Reusable Python scripts
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── ensemble_model.py
│   └── feature_engineering_ablation.py
├── models/
│   └── best_overall_model.pkl     # Trained Logistic Regression model
├── requirements.txt
├── README.md
├── Group_Project_Report.pdf       # ← Full report (PDF)
└── submission.csv                 # Ready for Kaggle upload
