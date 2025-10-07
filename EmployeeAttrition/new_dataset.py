import pandas as pd

# Load your original dataset
df = pd.read_csv(r"D:\attrition\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Sort by Department (optional)
df = df.sort_values("Department").reset_index(drop=True)

# Create EmployeeID column: start from 101 for each department
emp_ids = []

for dept in df["Department"].unique():
    dept_data = df[df["Department"] == dept]
    dept_count = dept_data.shape[0]
    # IDs start from 101 for this department
    emp_ids.extend(list(range(101, 101 + dept_count)))

df["EmployeeID"] = emp_ids

# Reorder columns to have EmployeeID first (optional)
cols = ["EmployeeID"] + [col for col in df.columns if col != "EmployeeID"]
df = df[cols]

# Save new dataset
df.to_csv("D:/attrition/WA_Fn-UseC_-HR-Employee-Attrition-WithID_new.csv", index=False)

# Count employees per department
dept_counts = df["Department"].value_counts()
print("Employee count per department:")
print(dept_counts)
