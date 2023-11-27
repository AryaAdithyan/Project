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
    forecasting_frequency = st.sidebar.radio("Select Forecasting Frequency", ["Hourly", "Daily", "Weekly","Monthly"])
    product_name = st.sidebar.selectbox("Select Product", ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"])
    num_intervals = st.sidebar.number_input("Enter Number of Intervals for Forecasting", min_value=1, max_value=30, value=7)

    if st.sidebar.button("Generate Forecast"):
        # Determine the dataset based on the selected frequency
        if forecasting_frequency == "Hourly":
            dataset_name = "saleshourly.csv"
        elif forecasting_frequency == "Daily":
            dataset_name = "salesdaily.csv"
        elif forecasting_frequency == "Weekly":
            dataset_name = "salesweekly.csv"
        elif  forecasting_frequency == "Monthly":
            dataset_name = "salesmonthly.csv"
        else:
            st.error("Invalid forecasting frequency selected")
            return

        # Load the specific dataset based on the user's selection
        file_path = dataset_name
        df = load_and_preprocess_data(file_path)

        # Train Auto-ARIMA model for short-term forecasting
        model_autoarima = auto_arima(df[product_name], seasonal=True, m=12)  # Adjust seasonality as needed
        model_autoarima.fit(df[product_name])

        # Generate future date range based on user input
        if forecasting_frequency == "Hourly":
            freq = "H"
        elif forecasting_frequency == "Daily":
            freq = "D"
        elif forecasting_frequency == "Weekly":
            freq = "W"
        elif forecasting_frequency == "Monthly":
            freq = "M"
        future_dates = pd.date_range(df.index[-1] + timedelta(hours=1), periods=num_intervals, freq=freq)

        # Predict sales for the future date range using Auto-ARIMA
        predictions = model_autoarima.predict(n_periods=num_intervals, return_conf_int=False)

        # Create a DataFrame for visualization
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Sales": predictions})
        forecast_df.set_index("Date", inplace=True)

        # Display the forecast
        st.subheader(f"Sales Forecast for {product_name} - {forecasting_frequency} Forecasting")
        st.line_chart(forecast_df)

if __name__ == "__main__":
    main()
