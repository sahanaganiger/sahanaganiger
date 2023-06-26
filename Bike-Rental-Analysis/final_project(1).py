# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:52:21 2023

@author: sachin
"""

import streamlit as st
import pandas as pd
import dill
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

dill_in = open('E://project//xgboost_model .pkl', 'rb')
reg = dill.load(dill_in)

def preprocess_data(data):
    encoder = LabelEncoder()
    data['season'] = encoder.fit_transform(data['season'])
    data['workingday'] = encoder.fit_transform(data['workingday'])
    return data

def predict(input_df):
    preprocessed_data = preprocess_data(input_df)
    preprocessed_data['hr'] = 0
    preprocessed_data['year'] = 0
    preprocessed_data['month'] = 0
    preprocessed_data['day'] = 0
    predictions = reg.predict(preprocessed_data)
    return predictions

def main():
    st.title("Bike Rental Calculation App")
    st.sidebar.header("Input Features")

    season = st.sidebar.selectbox("Season", ("spring", "summer", "fall", "winter"))
    mnth = st.sidebar.selectbox("Month", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    weekday = st.sidebar.selectbox("Weekday", (0, 1, 2, 3, 4, 5, 6))
    workingday = st.sidebar.selectbox("Workingday", ("No", "Yes"))
    temp = st.sidebar.slider("Temperature", 0, 100, 1) / 100
    atemp = temp
    hum = st.sidebar.slider("Humidity", 0, 100, 1) / 100
    casual = st.sidebar.number_input("Casual")
    registered = st.sidebar.number_input("Registered")
    input_data = pd.DataFrame({
        'season': [season],
        'mnth': [mnth],
        'weekday': [weekday],
        'workingday': [workingday],
        'temp': [temp],
        'atemp': [atemp],
        'hum': [hum],
        'casual': [casual],
        'registered': [registered]
    }, columns=['season', 'mnth', 'hr', 'weekday', 'workingday', 'temp', 'atemp', 'hum', 'casual', 'registered'])

    if st.button("Predict"):
        predictions = predict(input_data)
        st.success(f"The predicted bike rentals count is {predictions[0]:.0f}")

if __name__ == "__main__":
    main()
