# xgboost_employee_attrition.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# ==========================
# 1. Load dataset
# ==========================
df = pd.read_csv(r"D:\attrition\WA_Fn-UseC_-HR-Employee-Attrition.csv")  # update path if needed

# Target variable
y = df["Attrition"].map({"Yes": 1, "No": 0})  # convert Yes/No → 1/0
X = df.drop(columns=["Attrition"])

# Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ==========================
# 2. Train-test split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# 3. Preprocessing
# ==========================
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ==========================
# 4. Define XGBoost model
# ==========================
xgb_clf = xgb.XGBClassifier(
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,  # handle imbalance
    use_label_encoder=False,
    eval_metric="logloss"
)

# ==========================
# 5. Create pipeline (Preprocessing + SMOTE + Model)
# ==========================
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", xgb_clf)
])

# ==========================
# 6. Train model
# ==========================
pipeline.fit(X_train, y_train)

# ==========================
# 7. Predictions
# ==========================
train_pred = pipeline.predict(X_train)
test_pred = pipeline.predict(X_test)

train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
test_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# ==========================
# 8. Metrics
# ==========================
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
roc_auc = roc_auc_score(y_test, test_pred_proba)
pr_auc = average_precision_score(y_test, test_pred_proba)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy : {test_acc:.4f}")
print(f"ROC-AUC          : {roc_auc:.4f}")
print(f"PR-AUC           : {pr_auc:.4f}")

# ==========================
# 9. Plots
# ==========================
os.makedirs("reports", exist_ok=True)

# --- Training vs Testing Accuracy ---
plt.bar(["Train", "Test"], [train_acc, test_acc], color=["green", "blue"])
plt.title("Training vs Testing Accuracy - XGBoost")
plt.ylabel("Accuracy")
plt.savefig("reports/xgb_train_test_accuracy.png")
plt.close()

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, test_pred_proba)
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.savefig("reports/xgb_roc_curve.png")
plt.close()

# --- Precision-Recall Curve ---
prec, rec, _ = precision_recall_curve(y_test, test_pred_proba)
plt.plot(rec, prec, label=f"PR-AUC = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - XGBoost")
plt.legend()
plt.savefig("reports/xgb_pr_curve.png")
plt.close()

# ==========================
# 10. Save model
# ==========================
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/xgb_attrition_model.pkl")

print("\n✅ Model training complete. Results & plots saved.")
