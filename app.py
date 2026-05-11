import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

st.title("Customer Churn Prediction with Logistic Regression")

# Load data
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
st.subheader("Dataset Overview")
st.write("Shape:", data.shape)
st.write(data.head())

# Preprocessing
st.subheader("Data Preprocessing")
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()
st.write("After handling missing values:", data.shape)

# Encode categorical variables
encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Churn':
        data[col] = encoder.fit_transform(data[col])

# Encode target
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

st.write("Data types after encoding:")
st.write(data.dtypes)

# Features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.subheader("Train-Test Split")
st.write(f"Training data size: {len(X_train)}")
st.write(f"Testing data size: {len(X_test)}")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
st.subheader("Model Training")
st.write("Logistic Regression model trained successfully!")

# Model coefficients
st.write("Model Coefficients (m):", model.coef_[0][:5])  # Show first 5
st.write("Intercept (c):", model.intercept_[0])

# Predictions
y_pred = model.predict(X_test)

# Evaluation
st.subheader("Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")

# Classification report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# Prediction
st.subheader("Predict Customer Churn")
tenure = st.slider("Tenure", int(data['tenure'].min()), int(data['tenure'].max()), int(data['tenure'].mean()))
MonthlyCharges = st.slider("Monthly Charges", float(data['MonthlyCharges'].min()), float(data['MonthlyCharges'].max()), float(data['MonthlyCharges'].mean()))
TotalCharges = st.slider("Total Charges", float(data['TotalCharges'].min()), float(data['TotalCharges'].max()), float(data['TotalCharges'].mean()))

input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

for col in X.columns:
    if col not in input_data.columns:
        if data[col].dtype == 'object':
            input_data[col] = data[col].mode()[0]
        else:
            input_data[col] = data[col].mean()

input_data = input_data[X.columns]  

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[0][1]

st.write(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
st.write(f"Probability of Churn: {probability:.4f}")

if probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

