import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

DATA_FILE = "expenses.csv"
MODEL_FILE = "expense_predictor.joblib"

# Load data
if not os.path.exists(DATA_FILE):
    print("No expense data found! Please add some data in the app first.")
    exit()

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# Prepare monthly data
df["Month"] = df["Date"].dt.to_period("M")
monthly_expense = df.groupby("Month")["Amount"].sum().reset_index()

# Convert to numeric months for ML
monthly_expense["Month_Num"] = range(1, len(monthly_expense) + 1)

# Train model
X = monthly_expense[["Month_Num"]]
y = monthly_expense["Amount"]

model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, MODEL_FILE)

print("✅ Model trained and saved as 'expense_predictor.joblib'")
