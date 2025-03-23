import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# Load the trained model
model_xgb = joblib.load("xgboost_model.pkl")

# Load the dataset to get unique values for dropdowns
df = pd.read_csv("cleaned_data.csv")

# Ensure date is in datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce') 

# Extract unique values for dropdowns
unique_districts = df['district'].unique()
unique_markets = df['market'].unique()
unique_commodities = ['Rice (white)', 'Rice (medium grain)', 'Rice (red)', 'Rice (red nadu)'] 

# Streamlit UI
st.title("📊 Sri Lanka Rice Price Prediction Tool & Dashboard")

# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-image: url('https://img.freepik.com/free-photo/milled-rice-bowl-wooden-spoon-black-cement-floor_1150-20058.jpg');
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#         filter: brightness(100%); /* Adjust brightness (100% is default) */
#     }
#     </style>
#     """, 
#     unsafe_allow_html=True
# )

brightness = st.sidebar.slider("Adjust Background Brightness", 10, 200, 100)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('https://img.freepik.com/free-photo/milled-rice-bowl-wooden-spoon-black-cement-floor_1150-20058.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        filter: brightness({brightness}%);
    }}
    </style>
    """, 
    unsafe_allow_html=True
)


# User input section for prediction
st.sidebar.header("Input Data")

# Default to None so the prediction is not made until inputs are selected
year = st.sidebar.number_input("Select Year", min_value=2024, max_value=2030, value=None, step=1, format="%d")
month = st.sidebar.number_input("Select Month", min_value=1, max_value=12, value=None, step=1, format="%d")
day = st.sidebar.number_input("Select Day", min_value=1, max_value=31, value=None, step=1, format="%d")
district = st.sidebar.selectbox("Select District", [''] + list(unique_districts))
market = st.sidebar.selectbox("Select Market", [''] + list(unique_markets))
commodity = st.sidebar.selectbox("Select Commodity", [''] + unique_commodities)

# Show prediction only if all inputs are selected
if year and month and day and district and market and commodity:
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

    # One-hot encode categorical columns (district and market)
    new_data_encoded = pd.get_dummies(new_data, columns=['district', 'market'], drop_first=True)

    # Get feature names from the model's booster
    booster = model_xgb.get_booster()
    model_feature_names = booster.feature_names

    # Ensure encoded new data has same columns as model
    missing_cols = set(model_feature_names) - set(new_data_encoded.columns)
    for col in missing_cols:
        new_data_encoded[col] = 0

    # Reorder columns to match model
    new_data_encoded = new_data_encoded[model_feature_names]

    # Predict price
    predicted_price = model_xgb.predict(new_data_encoded)[0]

    # Display prediction
    st.subheader("📈 Predicted Rice Price")
    st.write(f"💰 Estimated Price: **Rs. {predicted_price:.2f}**")
else:
    st.subheader("📈 Predicted Rice Price")
    st.write("💰 Please enter all required inputs to get the estimated price.")

# Historical Price Trends Dashboard Section
st.subheader("📊 Rice Price Trends Dashboard")

# Display latest prices
st.subheader("📋 Latest Rice Prices")
st.dataframe(df.tail(10))

# Historical trends
if commodity:
    filtered_df = df[df['commodity'] == commodity]
    fig_trend = px.line(filtered_df, x="date", y="price", title=f"Price Trend for {commodity}")
    st.plotly_chart(fig_trend)

    # Summary statistics
    st.subheader("🧑‍🔬 Price Statistics")
    avg_price = filtered_df['price'].mean()
    min_price = filtered_df['price'].min()
    max_price = filtered_df['price'].max()

    st.write(f"📊 **Average Price**: Rs. {avg_price:.2f}")
    st.write(f"📉 **Minimum Price**: Rs. {min_price:.2f}")
    st.write(f"📈 **Maximum Price**: Rs. {max_price:.2f}")

# Price comparison over time
st.subheader("📈 Price Over Time")
fig_comparison = px.line(df, x="date", y="price", color="commodity", title="Price Comparison for Different Commodities Over Time")
st.plotly_chart(fig_comparison)

# Seasonal trends
st.subheader("📈 Rice Price by Month")
fig_seasonal = px.line(df.groupby('month')['price'].mean().reset_index(), x='month', y='price', title='Average Rice Price by Month')
st.plotly_chart(fig_seasonal)


st.subheader("📈 Rice Price Forecast")
# Price forecasting using ARIMA
df.set_index('date', inplace=True)
model = ARIMA(df['price'], order=(5,1,0))
model_fit = model.fit()
future_forecast = model_fit.forecast(steps=12)

forecast_df = pd.DataFrame({'date': pd.date_range(start=df.index[-1], periods=12, freq='M'), 'price': future_forecast})
fig_forecast = px.line(df, x=df.index, y="price")
fig_forecast.add_scatter(x=forecast_df['date'], y=forecast_df['price'], mode='lines', name='Forecasted Price')
st.plotly_chart(fig_forecast)
