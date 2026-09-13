
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Energy Consumption Forecaster", page_icon="⚡", layout="centered")

st.title("⚡ PJM West Hourly Energy Consumption Forecaster")
st.write("ExcelR Internship Project | Deep Learning LSTM Model Deployment")

@st.cache_resource
def get_model():
    return load_model('energy_lstm_model.h5')

model = get_model()

st.subheader("Input Recent Energy Demand")
recent_avg = st.number_input("Enter recent average MW consumption:", min_value=500.0, max_value=15000.0, value=5500.0)

if st.button("Forecast Next Hour Consumption"):
    # Reshape input into lookback window (48 timesteps)
    dummy_input = np.array([[recent_avg]] * 48).reshape(1, 48, 1)
    prediction_scaled = model.predict(dummy_input)
    
    # Reverse scaling approximation for visualization output
    predicted_mw = float(prediction_scaled[0][0] * 10000)
    
    st.success(f"Predicted Energy Consumption: **{predicted_mw:.2f} MW**")
