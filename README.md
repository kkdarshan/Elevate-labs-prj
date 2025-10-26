AI chat bot:

Features
- Real-time text-based chat  
- Context-aware replies using `DialoGPT`  
- Flask web interface for user interaction  
- Timestamped conversations  
- Easy to extend for custom datasets or APIs  

Install dependencies:
pip install -r requirements.txt
python app.py
http://127.0.0.1:5000/


Stock Price Prediction using LSTM:

Project Overview
This project predicts **future stock prices** using **Long Short-Term Memory (LSTM)** deep learning networks.  
It uses **historical stock data** from Yahoo Finance to learn market patterns and forecast trends.

Features
- Fetches live stock market data using `yfinance`
- Preprocesses data with scaling and sequence generation
- Trains an **LSTM neural network** on past price trends
- Plots **Actual vs Predicted** stock prices
- Supports integration of **technical indicators** (like Moving Average, RSI)

Install Dependencies
pip install -r requirements.txt
python stock_lstm.py

View the Results:
The script will generate:
Training and validation loss graphs
Actual vs Predicted stock price chart

Example Output
Epoch 1/50
loss: 0.0124 - val_loss: 0.0102

Model accuracy graph generated!
Predicted vs Actual Stock Prices displayed.













