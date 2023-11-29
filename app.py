import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['datum'] = pd.to_datetime(df['datum'])
    df.set_index('datum', inplace=True)
    df.sort_index(inplace=True)
    return df

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length].values
        label = data.iloc[i+seq_length]
        sequences.append(seq)
        target.append(label)
    return np.array(sequences), np.array(target)

# Streamlit app
def main():
    st.set_page_config(
        page_title="Pharma Sales Forecasting App",
        page_icon=":pill:",
        layout="wide",
    )

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

        # Extract the relevant product's sales data
        sales_data = df[product_name].values.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(sales_data)

        # Create sequences for LSTM
        seq_length = 10  # Adjust as needed
        sequences, target = create_sequences(pd.DataFrame(scaled_data), seq_length)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Train the LSTM model
        model.fit(sequences, target, epochs=50, batch_size=32, verbose=0)

        # Prepare input sequence for prediction
        input_seq = scaled_data[-seq_length:]
        input_seq = input_seq.reshape(1, seq_length, 1)

        # Make predictions using the trained LSTM model
        predicted_value_scaled = model.predict(input_seq)[0][0]

        # Inverse transform the predicted value to the original scale
        predicted_value = scaler.inverse_transform(np.array([[predicted_value_scaled]]))[0][0]

        # Display the forecast and predicted values for LSTM
        st.subheader(f"LSTM Predicted Sales for {product_name} on {prediction_date}:")
        st.write(predicted_value)

if __name__ == "__main__":
    main()
