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
    
    # Allow the user to enter the date for prediction
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

        # Generate forecast for the specific prediction date
        forecast_arima = model_arima_fit.get_forecast(steps=1)
        prediction_value_arima = forecast_arima.predicted_mean.loc[pd.Timestamp(prediction_date)]

        # Generate future date range for visualization
        if forecasting_frequency == "Hourly":
            freq = "H"
        elif forecasting_frequency == "Daily":
            freq = "D"
        elif forecasting_frequency == "Weekly":
            freq = "W"
        elif forecasting_frequency == "Monthly":
            freq = "M"
        future_dates = pd.date_range(df.index[-1] + timedelta(hours=1), periods=7, freq=freq)

        # Predict sales for the future date range using Auto-ARIMA
        predictions_autoarima = model_autoarima.predict(n_periods=7, return_conf_int=False)

        # Create DataFrame for visualization
        forecast_df_autoarima = pd.DataFrame({"Date": future_dates, "Predicted Sales (Auto-ARIMA)": predictions_autoarima})
        forecast_df_autoarima.set_index("Date", inplace=True)

        # Display the forecast and predicted values for ARIMA
        st.subheader(f"ARIMA Predicted Sales for {product_name} on {prediction_date}:")
        st.write(prediction_value_arima)

        # Display the Auto-ARIMA forecast and predicted values
        st.subheader(f"Auto-ARIMA Sales Forecast for {product_name} - {forecasting_frequency} Forecasting")
        st.line_chart(forecast_df_autoarima)

        st.subheader("Auto-ARIMA Predicted Values:")
        st.write(forecast_df_autoarima)

if __name__ == "__main__":
    main()
