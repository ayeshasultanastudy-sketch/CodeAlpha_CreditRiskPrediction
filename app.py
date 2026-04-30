import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("credit_model.pkl")

st.title("💳 Credit Risk Prediction System")
st.write("Enter applicant details below:")

age = st.number_input("Age", 18, 100)
sex = st.selectbox("Gender", ["male", "female"])
job = st.number_input("Job (0-3)", 0, 3)
housing = st.selectbox("Housing", ["own", "rent", "free"])
credit_amount = st.number_input("Credit Amount")
duration = st.number_input("Loan Duration")

if st.button("Predict Risk"):
    
    input_df = pd.DataFrame([[
        age,
        0 if sex == "male" else 1,
        job,
        0,
        credit_amount,
        duration
    ]], columns=["Age","Sex","Job","Housing","Credit amount","Duration"])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠ High Credit Risk")
    else:
        st.success("✅ Low Credit Risk")