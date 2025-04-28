import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Customer Churn Prediction')

# Input fields for numerical features
tenure = st.slider('Tenure (months)', 0, 72, 12)
monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 200.0, 50.0)
total_charges = st.number_input('Total Charges ($)', 0.0, 10000.0, 500.0)

# Input fields for categorical/binary features
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
gender = st.selectbox('Gender', ['Male', 'Female'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Prepare input data
input_data = pd.DataFrame({
    'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'gender_Male': [1 if gender == 'Male' else 0],
    'Partner_Yes': [1 if partner == 'Yes' else 0],
    'Dependents_Yes': [1 if dependents == 'Yes' else 0],
    'PhoneService_Yes': [1 if phone_service == 'Yes' else 0],
    'MultipleLines_No phone service': [1 if multiple_lines == 'No phone service' else 0],
    'MultipleLines_Yes': [1 if multiple_lines == 'Yes' else 0],
    'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
    'InternetService_No': [1 if internet_service == 'No' else 0],
    'OnlineSecurity_No internet service': [1 if online_security == 'No internet service' else 0],
    'OnlineSecurity_Yes': [1 if online_security == 'Yes' else 0],
    'OnlineBackup_No internet service': [1 if online_backup == 'No internet service' else 0],
    'OnlineBackup_Yes': [1 if online_backup == 'Yes' else 0],
    'DeviceProtection_No internet service': [1 if device_pr
