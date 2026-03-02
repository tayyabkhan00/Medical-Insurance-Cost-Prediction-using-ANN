# 🏥 Medical-Insurance-Cost-Prediction-using-ANN

### 📌 Project Overview

This project predicts medical insurance charges based on personal and health-related attributes using an Artificial Neural Network (ANN) built with PyTorch.

The model is deployed as an interactive Streamlit web application, allowing users to input details and get real-time insurance cost predictions.

### 🚀 Features

- Deep Learning regression model (ANN)
- End-to-end pipeline (preprocessing → training → deployment)
- Feature encoding (categorical → numerical)
- Feature scaling using StandardScaler
- Model evaluation using R² score
- Interactive Streamlit web interface
- Real-time prediction system

### 📊 Dataset

Dataset used: Medical Cost Personal Dataset (Kaggle)

## Features:

- Age
- Sex
- BMI
- Number of children
- Smoker (yes/no)
- Region
- Charges (Target variable)

### 🧠 Model Architecture
```</> code
Input Layer
↓
Linear (64) + ReLU
↓
Linear (32) + ReLU
↓
Linear (16) + ReLU
↓
Output Layer (1 neuron - Linear Activation)
```
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Evaluation Metric: R² Score

### 📈 Model Performance

- R² Score: ~0.80 – 0.88
- The model captures non-linear relationships between health factors and insurance cost.

### 🛠 Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

### 📂 Project Structure
```</> code
insurance_project/
│
├── insurance.csv
├── train_model.py
├── app.py
├── insurance_model.pth
├── scaler.pkl
└── README.md
```
