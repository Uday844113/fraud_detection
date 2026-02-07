import streamlit as st
import pandas as pd

st.title("🚨 Fraud Detection Dashboard")

# Load or create your alerts data
alerts = pd.DataFrame({
	"Alert": [0, 1, 0, 1, 0],
	"RiskLevel": ["Low", "High", "Low", "High", "Medium"]
})  # Replace with your actual data loading logic

st.metric("Total Transactions", len(alerts))
st.metric("Fraud Alerts", alerts["Alert"].sum())

st.subheader("High Risk Transactions")
st.dataframe(alerts[alerts["RiskLevel"] == "High"].head(10))