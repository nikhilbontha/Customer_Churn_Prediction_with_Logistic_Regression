import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)



st.title("Customer Churn Prediction using Logistic Classification")


data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.subheader("Dataset Overview")

st.write("Dataset Shape:", data.shape)

st.write(data.head())


st.subheader("Data Preprocessing")

# Convert TotalCharges into numeric
data['TotalCharges'] = pd.to_numeric(
    data['TotalCharges'],
    errors='coerce'
)

# Remove missing values
data = data.dropna()

# Remove customerID column
data = data.drop('customerID', axis=1)

st.write("Dataset Shape After Cleaning:", data.shape)



encoder = LabelEncoder()

categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

st.subheader("Encoded Dataset")

st.write(data.head())



X = data.drop('Churn', axis=1)

y = data['Churn']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

st.subheader("Train Test Split")

st.write("Training Data Size:", len(X_train))

st.write("Testing Data Size:", len(X_test))



model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

st.subheader("Model Training")

st.success("Logistic Regression Model Trained Successfully!")



st.subheader("Model Parameters")

st.write("First 5 Coefficients:")

st.write(model.coef_[0][:5])

st.write("Intercept:")

st.write(model.intercept_[0])



y_pred = model.predict(X_test)



st.subheader("Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

st.write(f"Accuracy Score: {accuracy:.4f}")

st.write(f"Precision Score: {precision:.4f}")

st.write(f"Recall Score: {recall:.4f}")



st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.write(report_df)



st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

st.pyplot(fig)


st.subheader("Predict Customer Churn")

tenure = st.slider(
    "Tenure",
    int(data['tenure'].min()),
    int(data['tenure'].max()),
    int(data['tenure'].mean())
)

monthly_charges = st.slider(
    "Monthly Charges",
    float(data['MonthlyCharges'].min()),
    float(data['MonthlyCharges'].max()),
    float(data['MonthlyCharges'].mean())
)

total_charges = st.slider(
    "Total Charges",
    float(data['TotalCharges'].min()),
    float(data['TotalCharges'].max()),
    float(data['TotalCharges'].mean())
)


input_data = pd.DataFrame(columns=X.columns)

for col in X.columns:

    if col == 'tenure':
        input_data.loc[0, col] = tenure

    elif col == 'MonthlyCharges':
        input_data.loc[0, col] = monthly_charges

    elif col == 'TotalCharges':
        input_data.loc[0, col] = total_charges

    else:
        input_data.loc[0, col] = data[col].mode()[0]


prediction = model.predict(input_data)

probability = model.predict_proba(input_data)[0][1]


st.subheader("Prediction Result")

if prediction[0] == 1:

    st.error("Customer is likely to Churn")

else:

    st.success("Customer is not likely to Churn")

st.write(f"Churn Probability: {probability:.4f}")