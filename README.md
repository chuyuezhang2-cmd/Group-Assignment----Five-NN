# Titanic Survival Prediction - Machine Learning Project

**Course**: Machine Learning for Business  
**Group Name**: Five-NN
**Group Topic**: From Data Exploration to Survival Prediction: Machine Learning on Titanic Passenger Data

## Project Overview

This project predicts passenger survival in the 1912 Titanic disaster using machine learning — a classic binary classification task (Survived: 0 = No, 1 = Yes).  

We built an end-to-end machine learning pipeline including:

- Data understanding & Exploratory Data Analysis (EDA)
- Comprehensive data preprocessing & feature engineering
- Multiple model training & comparison
- Hyperparameter tuning & rigorous validation
- Model interpretation, error analysis & final model selection

**Final selected model**: Logistic Regression  
**Validation performance** (5-fold CV):  
- Accuracy: 82.12%  
- F1-score: 0.754  
- AUC-ROC: 0.858  

These results outperform typical baseline Titanic models (~78–80%) and clearly demonstrate the strong influence of **gender**, **passenger class**, and engineered features.

## Repository Structure
titanic-survival-prediction/
├── data/
│   ├── train.csv               # Original Kaggle training set (891 rows)
│   └── test.csv                # Kaggle test set (418 rows)
├── notebooks/
│   ├── 1_EDA.ipynb             # Exploratory Data Analysis & visualizations
│   ├── 2_Preprocessing.ipynb   # Missing value handling, encoding, feature engineering
│   ├── 3_Modeling_Comparison.ipynb  # Multiple algorithms + hyperparameter tuning
│   └── 4_Final_Model_Evaluation.ipynb  # Logistic Regression, error analysis, interpretation
├── src/
│   ├── data_preprocessing.py   # Reusable preprocessing functions
│   └── modeling.py             # Model training & evaluation utilities
├── models/
│   └── final_logreg_model.pkl  # Trained final model (can be loaded)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── report-draft.docx           # Full project report (detailed version)


## Key Highlights

- **Strongest predictors**: Sex (female advantage), Pclass (higher class → better survival), Title, FamilySize, Age×Pclass interaction
- **Feature engineering**: Title extraction from Name, FamilySize, AgePclass, Fare per person, HasCabin flag
- **Compared 7 algorithms** with GridSearchCV + 5-fold CV
- **Detailed error analysis** (confusion matrix, FP/FN inspection, ROC curve)
- **Interpretability** prioritized — final model is simple, explainable and performs well

## Technologies Used

Python 3.9+
pandas, numpy, matplotlib, seaborn
scikit-learn (preprocessing, model selection, GridSearchCV)
Jupyter Notebook

## Full Report
Detailed project report (including problem statement, EDA figures, model comparison tables, limitations & future work) is available in report-draft.docx.
We welcome any feedback or questions!
