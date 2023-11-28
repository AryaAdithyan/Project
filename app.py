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
    # Add a title with some style
    st.image("https://pbs.twimg.com/media/DywhyJiXgAIUZej?format=jpg&name=medium", width=800)  # Adjust width as needed
    st.title("🌟 Pharma Sales Forecasting App 🌟")
    st.subheader("Make data-driven decisions for your pharmaceutical products!")

    # Set background color
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for user inputs
    forecasting_frequency = st.sidebar.radio("Select Forecasting Frequency", ["Hourly", "Daily", "Weekly", "Monthly"])
    product_name = st.sidebar.selectbox("Select Product", ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"])
    num_intervals = st.sidebar.slider("Select Number of Intervals for Forecasting", min_value=1, max_value=30, value=7)

    start_date = st.sidebar.date_input("Select Start Date", datetime.today() - timedelta(days=365))
    end_date = st.sidebar.date_input("Select End Date", datetime.today())

    if start_date >= end_date:
        st.error("End date must be after start date.")
        return

    # Style the button
    generate_button = st.sidebar.button("Generate Forecast", key="generate_button")
    if generate_button:
        st.sidebar.success("Forecast generated successfully!")

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

    # Filter data based on selected date range
    df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

    # Train Auto-ARIMA model for short-term forecasting
    model_autoarima = auto_arima(df_filtered[product_name], seasonal=True, m=12)  # Adjust seasonality as needed
    model_autoarima.fit(df_filtered[product_name])

    # Generate future date range based on user input
    freq_mapping = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M"}
    freq = freq_mapping.get(forecasting_frequency, "D")  # Default to Daily if not found
    future_dates = pd.date_range(df_filtered.index[-1] + timedelta(hours=1), periods=num_intervals, freq=freq)

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




