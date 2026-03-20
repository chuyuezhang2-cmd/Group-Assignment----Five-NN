# Titanic Survival Prediction  
**Machine Learning for Business Course Project**

**Group Name**: Five-NN  
**Topic**: From Data Exploration to Survival Prediction: Machine Learning on Titanic Passenger Data

## Project Overview

This project predicts passenger survival in the 1912 Titanic disaster — a classic **binary classification** task (Survived: 0 = No, 1 = Yes).

We built a complete end-to-end machine learning pipeline, including:

- Data understanding & **Exploratory Data Analysis (EDA)**
- Comprehensive **data preprocessing** & **feature engineering**
- Multi-model training & performance comparison
- Hyperparameter tuning with **GridSearchCV** + 5-fold cross-validation
- Model interpretation, error analysis & final model selection

**Final Model**: Logistic Regression  
**5-Fold CV Performance**:

- Accuracy: **82.12%**
- F1-score: **0.754**
- AUC-ROC: **0.858**

These results significantly outperform typical baseline Titanic models (~78–80%) and highlight the strong predictive power of **gender**, **passenger class (Pclass)**, and engineered features.

## Repository Structure

<details>
<summary>Click to expand folder structure</summary>

```text
titanic-survival-prediction/
├── data/
│   ├── train.csv               # Kaggle training set (891 rows)
│   └── test.csv                # Kaggle test set (418 rows)
├── notebooks/
│   ├── 1_EDA.ipynb                     # Exploratory analysis & visualizations
│   ├── 2_Preprocessing.ipynb           # Missing values, encoding, feature engineering
│   ├── 3_Modeling_Comparison.ipynb     # Multiple algorithms & hyperparameter tuning
│   └── 4_Final_Model_Evaluation.ipynb  # Final model, error analysis, interpretation
├── src/
│   ├── data_preprocessing.py           # Reusable preprocessing functions
│   └── modeling.py                     # Model training & evaluation utilities
├── models/
│   └── final_logreg_model.pkl          # Trained final Logistic Regression model
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── report-draft.docx                   # Complete project report

```

## Key Highlights

- **Most important features**: Sex (female survival advantage), Pclass, Title, FamilySize, Age×Pclass interaction
- **Feature engineering**: Title from Name, FamilySize, Age×Pclass, Fare per person, HasCabin flag
- Compared **7 algorithms** using GridSearchCV + stratified 5-fold CV
- Detailed **error analysis** (confusion matrix, false positives/negatives inspection, ROC curve)
- Prioritized **interpretability** — final model is simple, explainable, and performs strongly

## Technologies Used

- Python 3.9.7
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (preprocessing, model selection, GridSearchCV)
- Jupyter Notebook, Pycharm, VScode

## Full Project Report

The detailed report (including problem definition, EDA figures, model comparison tables, limitations, and future work) is available in  
relevant word.

Feedback and questions are welcome!

