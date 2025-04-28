import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Customer Churn Prediction')

# Input fields
tenure = st.slider('Tenure (months)', 0, 72, 12)
monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 200.0, 50.0)
total_charges = st.number_input('Total Charges ($)', 0.0, 10000.0, 500.0)
contract_month = st.selectbox('Contract Month-to-Month?', ['Yes', 'No'])

# Prepare input data
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract_Month-to-month': [1 if contract_month == 'Yes' else 0],
    # Add other features to match X.columns (simplified for now)
})

# Scale numerical features
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Predict
if st.button('Predict'):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]
    st.write(f'Churn Prediction: {"Yes" if prediction[0] == 1 else "No"}')
    st.write(f'Churn Probability: {prob:.2%}')
