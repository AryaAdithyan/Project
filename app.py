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
    st.image("https://indoreinstitute.com/wp-content/uploads/2019/12/An-Insight-into-the-Different-Types-of-Pharmaceutical-Formulations.jpg", width=800)  # Adjust width as needed
    st.title("💊Pharma Sales Forecasting App💊")
    st.subheader("Make data-driven decisions for your pharmaceutical products!")

    # Set background color and padding
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
            padding: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for user inputs
    forecasting_frequency = st.sidebar.radio("Select Forecasting Frequency", ["Hourly", "Daily", "Weekly", "Monthly"])
    product_name = st.sidebar.selectbox("Select Drug Category", ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"])
    num_intervals = st.sidebar.slider("Select Number of Intervals for Forecasting", min_value=1, max_value=50, value=7)

    start_date_str = st.sidebar.text_input("Start Date", (datetime.today() - timedelta(days=30)).strftime("%Y/%m/%d"))
    end_date_str = st.sidebar.text_input("End Date", datetime.today().strftime("%Y/%m/%d"))

    # Convert the strings to datetime objects
    start_date = pd.to_datetime(start_date_str, format="%Y/%m/%d")
    end_date = pd.to_datetime(end_date_str, format="%Y/%m/%d")

    if start_date >= end_date:
        st.error("End date must be after start date.")
        return

    # Style the button
    generate_button = st.sidebar.button("Generate Forecast", key="generate_button")

    # Load the specific dataset based on the user's selection
    file_path = "salesdaily.csv"  # Assuming daily frequency for this example
    df = load_and_preprocess_data(file_path)

    # Filter data based on selected date range
    df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

    # Check if the button is clicked
    if generate_button:
        # Check if the filtered DataFrame is empty
        if df_filtered.empty:
            st.error("No data available for the selected date range.")
        else:
            # Train Auto-ARIMA model for short-term forecasting
            model_autoarima = auto_arima(df_filtered[product_name], seasonal=True, m=12, D=1)  # Adjust seasonality as needed
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
            st.success("Forecast generated successfully!")

if __name__ == "__main__":
    main()
