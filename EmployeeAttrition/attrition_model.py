# ==============================
# Employee Attrition Prediction using Random Forest
# ==============================

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load dataset
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print("Dataset loaded successfully!")

# Step 3: Encode categorical variables
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
categorical_cols = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Step 4: Split features & target
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Step 5: Handle class imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"Shape before SMOTE: {X.shape}, Shape after SMOTE: {X_res.shape}")

# Step 6: Scale features
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Step 8: Train Random Forest Model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest model trained successfully!")

# Step 9: Predictions
y_pred = rf.predict(X_test)

# Step 10: Evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 12: Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

print("\nTop 10 Important Features:")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices[:10]], y=feature_names[indices[:10]])
plt.title("Top 10 Feature Importances")
plt.show()

# Step 13: Save the model
joblib.dump(rf, "employee_attrition_model.pkl")
print("Model saved as employee_attrition_model.pkl")
