"""
Compare multiple models and select the best one.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# 1. Load preprocessed data
data_path = os.path.join('..', 'data', 'preprocessed_train.csv')
df = pd.read_csv(data_path)

X = df.drop('Survived', axis=1)
y = df['Survived']
print("Feature shape:", X.shape)

# 2. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define models and parameter grids (simplified for speed)
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(verbose=-1, random_state=42),
    'Neural Network': MLPClassifier(max_iter=500, random_state=42)
}

param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 25)],
        'alpha': [0.0001, 0.001],
        'activation': ['relu']
    }
}

# 4. Train and evaluate each model
results = []
best_models = {}

for name in models.keys():
    print(f"\n--- Training {name} ---")
    model = models[name]
    param_grid = param_grids[name]

    # Use GridSearchCV with 5-fold CV
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    # Best model
    best_model = grid.best_estimator_
    best_models[name] = best_model

    # Validation metrics
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, "predict_proba") else None

    val_acc = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_prob) if y_prob is not None else None

    results.append({
        'Model': name,
        'Best Params': grid.best_params_,
        'CV Accuracy': grid.best_score_,
        'Val Accuracy': val_acc,
        'Val F1': val_f1,
        'Val AUC': val_auc
    })

    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

# 5. Display comparison table
results_df = pd.DataFrame(results)
print("\n\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# 6. Voting Ensemble
# Use the top 3 models (by CV accuracy) for voting
top_models = results_df.nlargest(3, 'CV Accuracy')['Model'].tolist()
estimators = [(name, best_models[name]) for name in top_models]

voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train, y_train)
y_pred_vote = voting_clf.predict(X_val)
y_prob_vote = voting_clf.predict_proba(X_val)[:, 1]
vote_acc = accuracy_score(y_val, y_pred_vote)
vote_f1 = f1_score(y_val, y_pred_vote)
vote_auc = roc_auc_score(y_val, y_prob_vote)

print("\n--- Voting Ensemble (soft) ---")
print(f"Val Accuracy: {vote_acc:.4f}, Val F1: {vote_f1:.4f}, Val AUC: {vote_auc:.4f}")

# Add to results
results_df.loc[len(results_df)] = {
    'Model': 'Voting Ensemble (top3)',
    'Best Params': '-',
    'CV Accuracy': None,
    'Val Accuracy': vote_acc,
    'Val F1': vote_f1,
    'Val AUC': vote_auc
}

# 7. Save best overall model
# Choose based on validation accuracy + F1
best_row = results_df.loc[results_df['Val Accuracy'].idxmax()]
best_model_name = best_row['Model']
if best_model_name == 'Voting Ensemble (top3)':
    best_model_obj = voting_clf
else:
    best_model_obj = best_models[best_model_name]

model_path = os.path.join('..', 'models', 'best_overall_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(best_model_obj, model_path)
print(f"\nBest overall model ({best_model_name}) saved to {model_path}")

val_results = pd.DataFrame({
    'True': y_val,
    'Predicted': voting_clf.predict(X_val),
    'Probability': voting_clf.predict_proba(X_val)[:, 1]
})
val_results.to_csv('../data/val_predictions.csv', index=False)
print("Validation predictions saved.")

# Feature importance analysis (if best model is tree-based)
import matplotlib.pyplot as plt
import seaborn as sns

# If the best model is Voting Ensemble and contains a tree-based model,
# we extract feature importance from one of them (Random Forest as an example)
rf_in_ensemble = False
for name, model in estimators:
    if name == 'Random Forest':
        rf_model = model
        rf_in_ensemble = True
        break

if rf_in_ensemble:
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print("\n=== Feature Importances (Random Forest in Ensemble) ===")
    print(feat_imp.head(10))

    # Plot feature importance bar chart
    plt.figure(figsize=(10, 6))
    feat_imp.head(10).plot(kind='bar')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    # Save figure to figures folder
    os.makedirs('../figures', exist_ok=True)
    plt.savefig('../figures/feature_importance.png', dpi=150)
    plt.show()
else:
    print("\nNo tree-based model found in ensemble for feature importance analysis.")

# Error analysis
y_pred_vote = voting_clf.predict(X_val)
y_prob_vote = voting_clf.predict_proba(X_val)[:, 1]

# Create DataFrame with error information
errors_df = X_val.copy()
errors_df['True'] = y_val.values
errors_df['Predicted'] = y_pred_vote
errors_df['Probability'] = y_prob_vote
errors_df['Correct'] = errors_df['True'] == errors_df['Predicted']

# Find misclassified samples
error_samples = errors_df[~errors_df['Correct']]
print("\n=== Error Analysis ===")
print(f"Total validation samples: {len(X_val)}")
print(f"Number of errors: {len(error_samples)}")
print(f"Error rate: {len(error_samples) / len(X_val):.2%}")

# Show first few error samples
key_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
available_key_features = [f for f in key_features if f in error_samples.columns]
print("\nSample of errors (first 5):")
print(error_samples[available_key_features + ['True', 'Predicted', 'Probability']].head())

# Analyze error types: false positives vs false negatives
fp = error_samples[(error_samples['True'] == 0) & (error_samples['Predicted'] == 1)]
fn = error_samples[(error_samples['True'] == 1) & (error_samples['Predicted'] == 0)]
print(f"\nFalse Positives (predicted survived but actually died): {len(fp)}")
print(f"False Negatives (predicted died but actually survived): {len(fn)}")

# Save error samples to CSV
error_samples.to_csv('../data/error_analysis.csv', index=False)
print("\nError samples saved to ../data/error_analysis.csv")

# Confusion matrix visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred_vote)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Died', 'Survived'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Voting Ensemble')
plt.tight_layout()
plt.savefig('../figures/confusion_matrix.png', dpi=150)
plt.show()