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
├── data/                         
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessed_train.csv
│   └── preprocessed_test.csv
├── notebooks/                    
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing.ipynb
│   ├── 3_Modeling_Comparison.ipynb
│   └── 4_Final_Model_Evaluation.ipynb
├── figures/                      
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── src/                          
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── ensemble_model.py
│   └── feature_engineering_ablation.py
├── models/
│   └── best_overall_model.pkl    
├── requirements.txt
├── README.md
└── Group-Assignment----Five-NN.pdf                
```markdown
## ⚙️ Setup and Installation (Reproducibility)

### Prerequisites
- Python 3.9.7 or higher
- Git

### Step-by-step

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Create virtual environment** (strongly recommended)
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install all dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Data**  
   `train.csv` and `test.csv` are already included in the `data/` folder — no download needed.

---

## ▶️ How to Run & Reproduce Results

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Run notebooks in exact order** (recommended):
   - `1_EDA.ipynb` → Data exploration
   - `2_Preprocessing.ipynb` → Preprocessing + feature engineering
   - `3_Modeling_Comparison.ipynb` → Hyperparameter tuning & model comparison
   - `4_Final_Model_Evaluation.ipynb` → Final evaluation, error analysis & submission

   **Alternative** (run scripts directly):
   ```bash
   python src/data_preprocessing.py
   python src/feature_engineering_ablation.py
   ```

---

## 🔄 Loading the Trained Model

```python
import pickle

with open('models/best_overall_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")
```

---

## 📤 Generating Kaggle Submission

The notebook `4_Final_Model_Evaluation.ipynb` automatically generates `submission.csv` in the root folder.  
Just upload this file directly to the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).

---

## 📘 Full Project Report

The complete **40-point group report** (including Team Information, Problem Statement, EDA, Preprocessing, Modeling, Evaluation, Limitations & References) is included as:

**`Group-Assignment----Five-NN.pdf`** (already placed in the repository root)

---

## 🛠️ Technologies Used

- **Python** 3.9.7
- **Core**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML**: scikit-learn (GridSearchCV, pipelines), xgboost, lightgbm
- **Environment**: Jupyter Notebook
- **Others**: pickle, requirements.txt

---

## 💬 Feedback

Questions, suggestions or issues? Feel free to open an issue or contact the team!

**Submission Checklist** ✅
- Folder name: `Group Assignment -- Five-NN`
- GitHub repo is public
- TA/Instructor can access all links

---

**Thank you for checking out our project!**  
We hope this clean, fully reproducible repository helps you understand our Titanic survival prediction pipeline.  
Good luck with grading! 🚀
