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
### ⚙️ Installation & Setup

## 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/insurance-ann-project.git
cd insurance-ann-project
```
## 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
## ▶️ Train the Model
```bash
python train_model.py
```
This will:
- Train the ANN model
- Save insurance_model.pth
- Save scaler.pkl

### 🌐 Run Streamlit App
```bash
streamlit run app.py
```
Open the browser link shown in terminal.

### 🖥️ Streamlit App Features
- Interactive sliders and dropdowns
- Real-time prediction
- Clean user interface
- Business-ready prototype

### 💡 Business Use Case
This system can help:
- Insurance companies estimate premium costs
- Customers estimate expected insurance expenses
- Healthcare analytics platforms build pricing systems

### 📚 Key Learnings

- Handling categorical variables with One-Hot Encoding
- Importance of feature scaling in neural networks
- Building custom ANN architecture in PyTorch
- Saving and loading trained models
- Deploying ML/DL models using Streamlit
- Creating end-to-end ML applications

### 📌 Future Improvements

- Hyperparameter tuning
- Add regularization (Dropout)
- Compare with Linear Regression / XGBoost
- Deploy on Streamlit Cloud
- Add visual analytics dashboard

### 👨‍💻 Author

Tayyab Khan<br>
B.Tech — AI & Data Science<br>
Aspiring Data Scientist & AI Engineer
