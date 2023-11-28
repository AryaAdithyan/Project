import streamlit as st
import pandas as pd
from pmdarima import auto_arima
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Function to load and preprocess data
@st.cache_data(allow_output_mutation=True)
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['datum'] = pd.to_datetime(df['datum'])
    df.set_index('datum', inplace=True)
    df.sort_index(inplace=True)
    return df

# Function to train LSTM model
@st.cache_resource(allow_output_mutation=True)
def train_lstm_model(data, n_steps=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_train, y_train = [], []
    for i in range(len(data) - n_steps):
        x_train.append(scaled_data[i : i + n_steps, 0])
        y_train.append(scaled_data[i + n_steps, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=50, batch_size=8)

    return model, scaler

# Function to train Auto-ARIMA model
@st.cache_resource(allow_output_mutation=True)
def train_autoarima_model(data):
    model_autoarima = auto_arima(data, seasonal=True, m=12)
    model_autoarima.fit(data)
    return model_autoarima

# Function to generate forecast
def generate_forecast(df, product_name, forecasting_frequency, num_intervals, start_date, end_date):
    # Train Auto-ARIMA model for short-term forecasting
    model_autoarima = train_autoarima_model(df[product_name])
    
    future_dates_arima = pd.date_range(df.index[-1] + timedelta(hours=1), periods=num_intervals, freq=freq)
    predictions_arima = model_autoarima.predict(n_periods=num_intervals, return_conf_int=False)
    forecast_df_arima = pd.DataFrame({"Date": future_dates_arima, "Predicted Sales (ARIMA)": predictions_arima})
    forecast_df_arima.set_index("Date", inplace=True)
    st.subheader(f"ARIMA Sales Forecast for {product_name} - {forecasting_frequency} Forecasting")
    st.line_chart(forecast_df_arima)
    st.subheader("ARIMA Predicted Values:")
    st.write(forecast_df_arima)
    

    # Train LSTM model for long-term forecasting
    data_lstm = df[product_name].values.reshape(-1, 1)
    model_lstm, scaler_lstm = train_lstm_model(data_lstm, n_steps=7)
    future_dates_lstm = pd.date_range(df.index[-1] + timedelta(days=1), periods=num_intervals, freq=freq)
    x_input = data_lstm[-7:].reshape((1, 7, 1))
    x_input_scaled = scaler_lstm.transform(x_input)
    lstm_prediction_scaled = model_lstm.predict(x_input_scaled, batch_size=1)
    lstm_prediction = scaler_lstm.inverse_transform(lstm_prediction_scaled)
    forecast_df_lstm = pd.DataFrame({"Date": future_dates_lstm, "Predicted Sales (LSTM)": lstm_prediction.flatten()})
    forecast_df_lstm.set_index("Date", inplace=True)
    st.subheader(f"LSTM Sales Forecast for {product_name} - {forecasting_frequency} Forecasting")
    st.line_chart(forecast_df_lstm)
    st.subheader("LSTM Predicted Values:")
    st.write(forecast_df_lstm)

# Streamlit app
def main():
    # Load data outside main function
    file_path = "saleshourly.csv"  # Adjust as needed
    df = load_and_preprocess_data(file_path)

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
    product_name = st.sidebar.selectbox("Select Product", ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"])
    num_intervals = st.sidebar.slider("Select Number of Intervals for Forecasting", min_value=1, max_value=30, value=7)

    start_date = pd.Timestamp(st.sidebar.date_input("Select Start Date", datetime.today() - timedelta(days=365)))
    end_date = pd.Timestamp(st.sidebar.date_input("Select End Date", datetime.today()))

    if start_date >= end_date:
        st.error("End date must be after start date.")
        return

    # Style the button
    generate_button = st.sidebar.button("Generate Forecast", key="generate_button")
    if generate_button:
        generate_forecast(df, product_name, forecasting_frequency, num_intervals, start_date, end_date)
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

   # Example: Add a header for Additional Visualization Section
st.header("Additional Visualization Section")

# Placeholder content for additional visualization section
st.write("This is the content for the additional visualization section.")

# Example: Add a header for User Interaction Section
st.header("User Interaction Section")

# Placeholder content for user interaction section
st.write("This is the content for the user interaction section.")


if __name__ == "__main__":
    main()



