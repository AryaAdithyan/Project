import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from datetime import datetime, timedelta

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['datum'] = pd.to_datetime(df['datum'])
    df.set_index('datum', inplace=True)
    df.sort_index(inplace=True)
    return df

# Streamlit app
def main():
    st.title("Pharma Sales Forecasting App")

    # Sidebar for user inputs
    forecasting_frequency = st.sidebar.radio("Select Forecasting Frequency", ["Hourly", "Daily", "Weekly", "Monthly"])
    product_name = st.sidebar.selectbox("Select Product", ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"])
    prediction_date = st.sidebar.date_input("Enter Date for Prediction", datetime.today() + timedelta(days=1))

    if st.sidebar.button("Generate Forecast"):
        # Determine the dataset based on the selected frequency
        if forecasting_frequency == "Hourly":
            dataset_name = "saleshourly.csv"
        elif forecasting_frequency == "Daily":
            dataset_name = "salesdaily.csv"
        elif forecasting_frequency == "Weekly":
            dataset_name = "salesweekly.csv"
        elif forecasting_frequency == "Monthly":
            dataset_name = "salesmonthly.csv"
        else:
            st.error("Invalid forecasting frequency selected")
            return

        # Load the specific dataset based on the user's selection
        file_path = dataset_name
        df = load_and_preprocess_data(file_path)

        # Train ARIMA model for short-term forecasting
        model_arima = ARIMA(df[product_name], order=(5, 1, 2))  # Adjust order as needed
        model_arima_fit = model_arima.fit()

        # Train Auto-ARIMA model for short-term forecasting
        model_autoarima = auto_arima(df[product_name], seasonal=True, m=12)  # Adjust seasonality as needed
        model_autoarima.fit(df[product_name])

        # Predict sales for the specified date using ARIMA
        prediction_arima = model_arima_fit.get_forecast(steps=1).predicted_mean.loc[prediction_date]

        # Predict sales for the specified date using Auto-ARIMA
        prediction_autoarima = model_autoarima.predict(n_periods=1).loc[prediction_date]

        # Display the forecasts for the specified date
        st.subheader(f"ARIMA Sales Forecast for {product_name} - {forecasting_frequency} Forecasting on {prediction_date}")
        st.write(f"Predicted Sales (ARIMA): {prediction_arima}")

        st.subheader(f"Auto-ARIMA Sales Forecast for {product_name} - {forecasting_frequency} Forecasting on {prediction_date}")
        st.write(f"Predicted Sales (Auto-ARIMA): {prediction_autoarima}")

if __name__ == "__main__":
    main()
