import pandas as pd
import joblib

# Load dataset and trained model
df = pd.read_csv(r"D:\attrition\WA_Fn-UseC_-HR-Employee-Attrition-WithID_new.csv")
model = joblib.load(r"D:\attrition\models\xgb_attrition_model_new.pkl")  # trained XGBoost pipeline

def predict_employee_attrition():
    """
    Predict attrition for an employee based on EmployeeID and Department.
    """
    # Input EmployeeID and Department
    emp_id = input("Enter Employee ID: ")
    department = input("Enter Department: ")

    # Filter employee
    emp_data = df[(df["EmployeeID"] == int(emp_id)) & (df["Department"] == department)]

    if emp_data.empty:
        print(f"Employee ID {emp_id} in {department} not found.")
        return

    # Show all employee details
    print("\nEmployee Details:")
    print(emp_data.to_string(index=False))

    # Ask if user wants to predict
    choice = input("\nDo you want to predict attrition for this employee? (yes/no): ").lower()
    if choice != "yes":
        print("Prediction cancelled.")
        return

    # Drop target column for prediction
    emp_features = emp_data.drop(columns=["Attrition", "EmployeeID"])

    # Predict
    pred_class = model.predict(emp_features)[0]
    pred_proba = model.predict_proba(emp_features)[:, 1][0]

    # Map prediction to Yes/No
    pred_label = "Yes" if pred_class == 1 else "No"

    # Show prediction
    print("\n✅ Attrition Prediction:")
    print(f"Attrition Predicted: {pred_label}")
    print(f"Probability of leaving: {pred_proba:.2f}")

# Example usage
predict_employee_attrition()
