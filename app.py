import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

# Load scaler
scaler = joblib.load("scaler.pkl")

# Model structure
class ANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# Input dimension = 8 (after encoding)
model = ANN(8)
model.load_state_dict(torch.load("insurance_model.pth"))
model.eval()

st.title("Medical Insurance Cost Prediction")

age = st.slider("Age", 18, 100)
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.slider("Children", 0, 5)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "northeast", "southeast", "southwest"])

# Convert to dataframe
input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex=="male" else 0,
    "smoker_yes": 1 if smoker=="yes" else 0,
    "region_northwest": 1 if region=="northwest" else 0,
    "region_southeast": 1 if region=="southeast" else 0,
    "region_southwest": 1 if region=="southwest" else 0,
}

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)
input_tensor = torch.FloatTensor(input_scaled)

if st.button("Predict"):
    prediction = model(input_tensor)
    st.success(f"Estimated Insurance Cost: ${prediction.item():,.2f}")