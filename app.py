import streamlit as st
import pickle
import numpy as np

scaler = pickle.load(open('scaler4.pkl', 'rb'))
model = pickle.load(open("model1.pkl", 'rb'))

st.title("📈 Facebook Ads Conversion Predictor")

sms = st.text_input("Enter number of clicks and views (comma-separated)")

if st.button("Predict"):
    try:
        # Expect input like: "300,1200"
        input_vals = np.array([float(i) for i in sms.split(",")]).reshape(1, -1)
        scaled = scaler.transform(input_vals)
        result = model.predict(scaled)
        st.success(f"Predicted Conversions: {int(result[0])}")
    except Exception as e:
        st.error("❌ Invalid input. Please enter two numbers separated by a comma.")
