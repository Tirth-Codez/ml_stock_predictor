# app.py - The Python Backend for our ML Stock Predictor (Updated)

# --- 1. Import Necessary Libraries ---
from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS # To handle requests from the browser
import yfinance as yf # To fetch stock data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 2. Initialize the Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- 3. Global Variables & Model Setup ---
model = None
scaler = MinMaxScaler(feature_range=(0, 1))

def build_and_train_model(data):
    """
    A simplified function to build and train our LSTM model on the fly.
    This is for demonstration purposes.
    """
    global model, scaler
    
    # Prepare the dataset
    dataset = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(dataset)

    # Create training data sequences
    prediction_days = 60
    x_train, y_train = [], []

    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile and "train" the model (with a few steps for speed)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=5)

# --- 4. Define the API Endpoint ---
@app.route('/')
def home():
    """Serves the frontend HTML file."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_stock():
    """
    The main API endpoint that receives a ticker and optional buyPrice,
    fetches data, and returns a contextual prediction.
    """
    global model

    json_data = request.get_json()
    if not json_data or 'ticker' not in json_data:
        return jsonify({'error': 'Ticker not provided'}), 400

    ticker_input = json_data['ticker']
    buy_price = json_data.get('buyPrice') # Safely get buyPrice, might be None

    # Sanitize input to get a single ticker
    ticker = ticker_input.split(',')[0].split(' ')[0].strip()
    if not ticker:
        return jsonify({'error': 'Ticker symbol is invalid.'}), 400
    
    try:
        # Fetch data
        print(f"Fetching data for sanitized ticker: {ticker}")
        data = yf.download(ticker, period="200d", interval="1d")
        
        if data.empty:
            return jsonify({'error': f'Could not fetch data for ticker: "{ticker}". Please check if it is a valid symbol.'}), 404

        if isinstance(data['Close'], pd.DataFrame):
            print("Warning: yfinance returned a multi-ticker format. Selecting the first ticker's data.")
            first_ticker_symbol = data.columns.levels[1][0]
            data = data.xs(first_ticker_symbol, level=1, axis=1)

        # Train model and make prediction
        print(f"Building and training a temporary model for {ticker}...")
        build_and_train_model(data)
        print("Model training complete.")

        prediction_days = 60
        close_prices = data['Close'].values.reshape(-1, 1)
        scaled_inputs = scaler.transform(close_prices)
        
        last_60_days_scaled = scaled_inputs[-prediction_days:]
        x_test = np.array([last_60_days_scaled])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_price_scaled = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)

        current_price = data['Close'].iloc[-1]
        prediction = float(predicted_price[0][0])
        
        # --- NEW: Contextual Recommendation Logic ---
        recommendation = ""
        reason = ""
        response_data = {
            'currentPrice': round(current_price, 2),
            'predictedPrice': round(prediction, 2)
        }

        if buy_price is not None:
            # Logic for users who OWN the stock
            buy_price = float(buy_price)
            profit_loss = ((current_price - buy_price) / buy_price) * 100
            response_data['buyPrice'] = buy_price
            response_data['profitLoss'] = round(profit_loss, 2)

            if prediction < current_price * 0.98: # Predicts >2% drop
                recommendation = "CONSIDER SELLING"
                if profit_loss > 0:
                    reason = f"The model predicts a drop to ₹{prediction:.2f}. You might consider selling to lock in your {profit_loss:.2f}% profit."
                else:
                    reason = f"The model predicts a drop to ₹{prediction:.2f}. You might consider selling to prevent further losses."
            elif prediction > current_price * 1.02: # Predicts >2% rise
                 recommendation = "HOLD"
                 if profit_loss > 0:
                     reason = f"The model predicts a rise to ₹{prediction:.2f}. Hold to potentially increase your gains."
                 else:
                     reason = f"The model predicts a rise to ₹{prediction:.2f}. Hold for potential price recovery."
            else:
                recommendation = "HOLD"
                reason = f"The model predicts a stable price. Your current position is a {profit_loss:.2f}% gain/loss. Hold and monitor."
        else:
            # Original logic for users who DO NOT own the stock
            if prediction > current_price * 1.01:
                recommendation = "BUY"
                reason = f"The model predicts a potential rise to ₹{prediction:.2f} from the current price."
            elif prediction < current_price * 0.99:
                recommendation = "SELL"
                reason = f"The model predicts a potential drop to ₹{prediction:.2f}. Consider avoiding or shorting."
            else:
                recommendation = "HOLD"
                reason = f"The model predicts a stable price. Wait for a clearer signal."
        
        response_data['recommendation'] = recommendation
        response_data['reason'] = reason

        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An internal error occurred. Check the server logs for details.'}), 500

# --- 5. Run the Server ---
if __name__ == '__main__':
    print("Starting Flask server... Please wait for the model to initialize.")

    app.run(debug=True, port=5000)
