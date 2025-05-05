import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load and prepare data
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🏥 Diabetes Progression Prediction App")
st.write("Enter the values below to predict diabetes disease progression:")

# Create input sliders for all features
user_input = {}
for feature in diabetes.feature_names:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    user_input[feature] = st.slider(
        f"{feature}", min_val, max_val, mean_val
    )

# Convert user input to dataframe
input_df = pd.DataFrame([user_input])

# Predict on user input
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"📊 Predicted Diabetes Progression Score: **{prediction:.2f}**")

# Optional: Show raw input
with st.expander("🔍 Show input data"):
    st.write(input_df)
