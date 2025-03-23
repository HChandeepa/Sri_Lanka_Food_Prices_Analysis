import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# Load the trained model
model_xgb = joblib.load("xgboost_model.pkl")

# Load the dataset to get unique values for dropdowns
df = pd.read_csv("cleaned_data.csv")

# Ensure date is in datetime format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')

# Extract unique values for dropdowns
unique_districts = df['district'].unique()
unique_markets = df['market'].unique()
unique_commodities = ['Rice (white)', 'Rice (medium grain)', 'Rice (red)', 'Rice (red nadu)']

# Streamlit UI
st.title("📊 Sri Lanka Food Price Prediction Tool")

# User input section
st.sidebar.header("Input Data")

year = st.sidebar.number_input("Select Year", min_value=2024, max_value=2030, value=2025, step=1)
month = st.sidebar.number_input("Select Month", min_value=1, max_value=12, value=2, step=1)
day = st.sidebar.number_input("Select Day", min_value=1, max_value=31, value=10, step=1)
district = st.sidebar.selectbox("Select District", unique_districts)
market = st.sidebar.selectbox("Select Market", unique_markets)
commodity = st.sidebar.selectbox("Select Commodity", unique_commodities)

# One-hot encoding for selected commodity
commodity_features = {f"commodity_{c}": 0 for c in unique_commodities}
commodity_features[f"commodity_{commodity}"] = 1

# Prepare input data for prediction
new_data = pd.DataFrame({
    'year': [year],
    'month': [month],
    'day': [day],
    **commodity_features,
    'district': [district],
    'market': [market]
})

# Predict price
predicted_price = model_xgb.predict(new_data)[0]

# Display prediction
st.subheader("📈 Predicted Food Price")
st.write(f"💰 Estimated Price: **Rs. {predicted_price:.2f}**")

# Plot historical trend
st.subheader("📊 Historical Price Trends")
filtered_df = df[df['commodity'] == commodity]
fig = px.line(filtered_df, x="date", y="price", title=f"Price Trend for {commodity}")
st.plotly_chart(fig)

# Show recent data
st.subheader("📋 Latest Food Prices")
st.dataframe(df.tail(10))
