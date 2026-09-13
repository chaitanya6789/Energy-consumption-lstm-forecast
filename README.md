# ⚡ PJM West Hourly Energy Consumption Forecasting & Deployment

This project features an end-to-end deep learning time-series forecasting web application built to predict hourly energy consumption (MW) for the PJM West region. Developed for the ExcelR Data Science and AI Internship, this solution utilizes a Recurrent Neural Network with Long Short-Term Memory (RNN-LSTM) architecture optimized with the Adam optimizer to capture complex sequential trends from over 143,000 hourly records. 

## 🚀 Key Features
* **Data Preprocessing & EDA:** Cleaned 143k+ rows[cite: 3], handled missing hourly intervals[cite: 3], and applied linear interpolation[cite: 3].
* **Deep Learning Model:** Built a GPU-accelerated LSTM network with a 48-hour sliding-window lookback horizon.
* **Production Deployment:** Integrated the trained `.h5` model into an interactive web application using Streamlit, deployed live via GitHub.

## 🛠️ Tech Stack
* **Python** (Pandas, NumPy, Scikit-Learn)
* **Deep Learning:** TensorFlow / Keras (LSTM)
* **Deployment:** Streamlit Cloud, GitHub
