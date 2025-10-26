# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import os
import ta

st.title("📈 Stock Price Trend Prediction (LSTM)")

ticker = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):
    # 1️⃣ Fetch stock data
    data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
    if data.empty:
        st.error("No data found for this symbol!")
        st.stop()

    # 2️⃣ Ensure 1D Close
    close_1d = data['Close'].squeeze()

    # 3️⃣ Add indicators
    data['MA20'] = close_1d.rolling(window=20).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close_1d, window=14).rsi()
    data = data.fillna(method='bfill')

    # 4️⃣ Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'MA20', 'RSI']])

    # 5️⃣ Prepare LSTM input
    time_steps = 60
    X = []
    y = []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    # 6️⃣ Train model if it doesn't exist
    model_file = "lstm_stock_model.h5"
    if os.path.exists(model_file):
        model = load_model(model_file)
        st.info("Loaded existing model")
    else:
        st.info("Training new LSTM model... Please wait")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, batch_size=32, epochs=10, verbose=1)  # Reduced epochs for speed
        model.save(model_file)
        st.success("Model trained and saved!")

    # 7️⃣ Make predictions
    predictions = model.predict(X)
    pred_full = np.zeros((len(predictions), scaled_data.shape[1]))
    pred_full[:, 0] = predictions[:, 0]
    predictions = scaler.inverse_transform(pred_full)[:, 0]

    # 8️⃣ Plot actual vs predicted
    st.subheader(f"Predicted vs Actual Prices for {ticker}")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(data['Close'].iloc[-len(predictions):].values, label="Actual", color='blue')
    ax.plot(predictions, label="Predicted", color='red')
    ax.legend()
    st.pyplot(fig)
