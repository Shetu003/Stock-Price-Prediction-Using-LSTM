import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Set the page configuration
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈")

# Initialize session state for storing data
if "data" not in st.session_state:
    st.session_state.data = None

# App title
st.title("📈 Stock Price Prediction Using LSTM")

# Sidebar for user inputs
st.sidebar.header("Enter Stock Details")
stock_ticker = st.sidebar.text_input("Stock Ticker Symbol (e.g., AAPL, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Load stock data
if st.sidebar.button("Load Data"):
    with st.spinner("Fetching stock data..."):
        try:
            data = yf.download(stock_ticker, start=start_date, end=end_date)
            if data.empty:
                st.error("No data found. Please check the ticker symbol or date range.")
            else:
                st.success(f"Stock data for {stock_ticker} loaded successfully!")
                st.session_state.data = data  # Save data in session state
                st.write(data.tail())
                st.line_chart(data['Close'], use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Train the model and predict
if st.sidebar.button("Predict Stock Prices"):
    if st.session_state.data is not None:
        st.subheader("Stock Price Prediction")
        data = st.session_state.data  # Retrieve the data from session state
        
        # Prepare the data
        close_prices = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Create sequences
        def create_sequences(data, time_step=60):
            X, y = [], []
            for i in range(time_step, len(data)):
                X.append(data[i - time_step:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot predictions vs actual
        st.write("### Actual vs Predicted Stock Prices")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test_actual, label="Actual Prices", color="blue")
        ax.plot(predictions, label="Predicted Prices", color="red")
        ax.set_title(f"{stock_ticker} Stock Price Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Please load data first!")
