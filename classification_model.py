import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt

df = pd.read_parquet('green_tripdata_2021-01_cleaned.parquet')

df = df[(df['payment_type'] == 1) | (df['payment_type'] == 2)].copy()
df['payment_type'] = (df['payment_type'] == 1).astype(int)

X = df[['trip_distance', 'passenger_count', 'RatecodeID', 'extra']].copy()
y = df['payment_type'].copy()

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"Train set: {X_train.shape[0]}")
print(f"Validation set: {X_val.shape[0]}")
print(f"Test set: {X_test.shape[0]}")

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_val = model.predict_proba(X_val)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

print("\n===== TRAINING SET =====")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Precision: {precision_score(y_train, y_pred_train):.4f}")
print(f"Recall: {recall_score(y_train, y_pred_train):.4f}")
print(f"F1-Score: {f1_score(y_train, y_pred_train):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_train, y_pred_proba_train):.4f}")

print("\n===== VALIDATION SET =====")
print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_val):.4f}")
print(f"Recall: {recall_score(y_val, y_pred_val):.4f}")
print(f"F1-Score: {f1_score(y_val, y_pred_val):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba_val):.4f}")

print("\n===== TEST SET =====")
acc_test = accuracy_score(y_test, y_pred_test)
prec_test = precision_score(y_test, y_pred_test)
rec_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)

print(f"Accuracy: {acc_test:.4f}")
print(f"Precision: {prec_test:.4f}")
print(f"Recall: {rec_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")
print(f"ROC-AUC: {roc_auc_test:.4f}")

print("\n===== CONFUSION MATRIX (Test Set) =====")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print(f"True Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives: {cm[1, 1]}")

print("\n===== MODEL COEFFICIENTS =====")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.6f}")
print(f"Intercept: {model.intercept_[0]:.6f}")

joblib.dump(model, 'classification_model.pkl')
print("\n✓ Model saved to classification_model.pkl")
