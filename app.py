import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="💰 Expense Tracker", layout="wide")
st.title("🧮 Personal Expense Tracker")

DATA_FILE = "expenses.csv"

# -------------------------------
# Load existing data
# -------------------------------
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["Date", "Category", "Description", "Amount"])

# -------------------------------
# Add new expense entry
# -------------------------------
st.subheader("➕ Add New Expense")

col1, col2, col3, col4 = st.columns(4)

with col1:
    date = st.date_input("Date", datetime.now())
with col2:
    category = st.selectbox("Category", ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"])
with col3:
    description = st.text_input("Description", "")
with col4:
    amount = st.number_input("Amount (₹)", min_value=0.0, step=0.01)

if st.button("Add Expense"):
    new_data = pd.DataFrame({
        "Date": [date],
        "Category": [category],
        "Description": [description],
        "Amount": [amount]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    st.success("✅ Expense added successfully!")

# -------------------------------
# Display expense table
# -------------------------------
st.subheader("📋 Expense History")

if df.empty:
    st.info("No expenses added yet.")
else:
    st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)

    # -------------------------------
    # Summary
    # -------------------------------
    st.subheader("📊 Expense Summary")

    total_spent = df["Amount"].sum()
    st.metric("Total Spent (₹)", f"{total_spent:,.2f}")

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Monthly Summary
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    monthly_summary = df.groupby("Month")["Amount"].sum().reset_index()

    # Plot monthly expenses
    fig, ax = plt.subplots()
    ax.plot(monthly_summary["Month"], monthly_summary["Amount"], marker="o")
    ax.set_title("Monthly Spending Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (₹)")
    ax.grid(True)
    st.pyplot(fig)

    # Category breakdown
    cat_summary = df.groupby("Category")["Amount"].sum().reset_index()

    fig2, ax2 = plt.subplots()
    ax2.pie(cat_summary["Amount"], labels=cat_summary["Category"], autopct="%1.1f%%", startangle=90)
    ax2.set_title("Expenses by Category")
    st.pyplot(fig2)
# -------------------------------
# 🔮 Predict next month expense using trained model
# -------------------------------
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

if os.path.exists("expense_predictor.joblib"):
    model = joblib.load("expense_predictor.joblib")
    last_month_num = monthly_summary.shape[0]
    next_month_num = np.array([[last_month_num + 1]])
    predicted_expense = model.predict(next_month_num)[0]

    st.subheader("🔮 Next Month Expense Prediction")
    st.metric("Predicted Next Month Spend (₹)", f"{predicted_expense:,.2f}")
else:
    st.info("⚙️ Train the model first by running: `python train_model.py`")
import pandas as pd
predicted_expense = model.predict(pd.DataFrame({"Month_Num": [last_month_num + 1]}))[0]
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
