# app/routes.py
from datetime import datetime
from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
import joblib
from .redis_helpers import get_predictions_from_cache, save_predictions_to_cache
from sklearn.impute import SimpleImputer
from .automate_commodity_forecasting import  get_prediction_by_date
from .services import fetch_commodity_data, validate_prediction


main = Blueprint('main', __name__)

# Route to fetch commodity data
@main.route("/fetch_data", methods=["GET"])
def fetch_data():
    symbol = request.args.get("symbol", "GC=F")  # Default is Gold (GC=F)
    data = fetch_commodity_data(symbol)
    return jsonify(data.tail().to_dict())  # Return the last 5 rows of the data

# Load default models (for demonstration, you can load others as needed)
 

# Function to get prediction based on selected date using the specified model
def get_prediction_by_date_main(selected_date, model):

        # Make prediction using the selected model
    if model == 'random_forest':

        random_forest_model = joblib.load('commodity_price_predictor.pkl')
        print("Model loaded successfully") 
        # Load the dataset
        df = pd.read_csv('commodity_price_data.csv')
        df = df.drop_duplicates()

        # Convert the 'date' column and selected_date to datetime format
        df['date'] = pd.to_datetime(df['date'])
        selected_date = pd.to_datetime(selected_date)

        # Find the latest date in the dataset
        latest_date = df['date'].max()

        # Ensure selected_date is earlier or equal to latest_date
        if selected_date > latest_date:
            print(f"Selected date {selected_date} is later than the latest available date {latest_date}. Using the latest date.")
            selected_data = df[df['date'] == latest_date]
        else:
            selected_data = df[df['date'] == selected_date]
        selected_data = selected_data.drop_duplicates()
        # Debug: Check if selected_data contains any rows
        if selected_data.empty:
            print(f"No data found for the selected date: {selected_date}")
        else:
            print("Selected data loaded successfully:")
            
            print(selected_data.head()) 

    
        assert selected_data.shape[0] <= 1, f"Expected at most one row, but got {selected_data.shape[0]} rows."
             
            


        # Perform the feature engineering steps (same as your training data)
        selected_data['daily_return'] = selected_data['close_price'].pct_change()
        selected_data['price_range'] = selected_data['high_price'] - selected_data['low_price']
        selected_data['5_day_MA'] = selected_data['close_price'].rolling(window=5).mean()
        selected_data['20_day_MA'] = selected_data['close_price'].rolling(window=20).mean()
        selected_data['momentum'] = selected_data['close_price'] - df['close_price'].shift(5)

        selected_data['RSI'] = calculate_rsi(selected_data)
        selected_data['volume_change'] = selected_data['volume'].pct_change()
        selected_data['VWAP'] = (selected_data['close_price'] * selected_data['volume']).cumsum() / selected_data['volume'].cumsum()
        if selected_date > latest_date:
            # Rolling Averages with min_periods
            selected_data['20_day_MA'] = selected_data['close_price'].rolling(window=20, min_periods=1).mean()

            # Momentum calculation with fillna for initial rows
            selected_data['momentum'] = selected_data['close_price'] - selected_data['close_price'].shift(5)
            selected_data['momentum'] = selected_data['momentum'].fillna(0)

            # RSI calculation with fallback
            selected_data['RSI'] = calculate_rsi(selected_data).fillna(0)

            # Volume percentage change
            selected_data['volume_change'] = selected_data['volume'].pct_change().fillna(0)

            # VWAP (if needed, calculate explicitly)
            selected_data['VWAP'] = (selected_data['close_price'] * selected_data['volume']).cumsum() / selected_data['volume'].cumsum()

        else:
            selected_data.bfill()
        print(f"selected_data------------->{selected_data}")
        
        features_to_scale = ['price_range', 'volume_change', 'momentum', 'daily_return', 
                            '5_day_MA', '20_day_MA', 'RSI', 'VWAP']
        
        
       
        prediction_scaled = random_forest_model.predict(selected_data[features_to_scale])
    
        
        actual_price = selected_data[selected_data['date'] == selected_date]['close_price'].values[0]
        error = prediction_scaled[0] - actual_price

        # Calculate the absolute error
        absolute_error = abs(error)
        percentage_error = (absolute_error / actual_price) * 100
        print(percentage_error)
        # Ensure selected_date is a string or datetime, and parse it
        if isinstance(selected_date, str):
            selected_date_str = datetime.strptime(selected_date, '%Y-%m-%d')
        elif isinstance(selected_date, datetime):
            selected_date_str = selected_date
        else:
            raise ValueError("selected_date must be a string or datetime object")

        # Ensure prediction_scaled is not None and has at least one element
        if prediction_scaled is not None and len(prediction_scaled) > 0:
            # Check if selected date is in the future
            if selected_date_str > datetime.now():
                return {
                    'predicted_price': prediction_scaled[0],
                    'absolute_error': "Actual data doesn't exist",
                    'source': model
                }
            else:

                # Return the predicted value and the error metric for past dates
                return {
                    'predicted_price': prediction_scaled[0],
                    'absolute_error': percentage_error,
                    'source': model
                }
        else:
            return {
                'predicted_price': 'Model will rebuild to give this result in future',
                'absolute_error': 'NA',
                'source': model
            }
        
    elif model == "sarimax_model":
        selected_date_str= datetime.strptime(selected_date, '%Y-%m-%d')
        # Get the prediction for the selected date
        predicted_price = get_prediction_by_date(selected_date)

        if predicted_price is not None:
            # If the selected date is in the future, return only predicted_price
            if selected_date_str > datetime.now():
                return {
                    'predicted_price': predicted_price,
                    'absolute_error': "Actual Data Doesnt exist",
                    'source': model
                }
            

            # If the selected date is in the past (or a trained date), validate prediction
            absolute_error = validate_prediction(selected_date_str, predicted_price)

            # Return the predicted value and the error metric for past dates
            return {
                'predicted_price': predicted_price,
                'absolute_error': absolute_error,
                'source': model
            }
        else:
            return {
                'predicted_price': 'Model will rebuild to give this result in future',
                'absolute_error': 'NA',
                'source': model
            }
    elif model == "lstm":
        response = predict_future_price(selected_date)
        return {
                'predicted_price': response,
                'absolute_error': 'NA',
                'source': model
            }


# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the selected date from the request
        selected_date = request.json.get('date')
        model_type = request.json.get('model_type', 'random_forest')
        print(f"Selected Date: {selected_date}")
        response =   get_prediction_by_date_main(selected_date, model_type)
        print(response)
        return jsonify({
                        'predicted_price': response["predicted_price"],
                        'absolute_error': response["absolute_error"],
                        'source': response["source"]
                    }), 200
            

    except Exception as e:
        return jsonify({'error': str(e)}), 500





import yfinance as yf
import numpy as np
import joblib
from tensorflow.keras.models import load_model


def predict_future_price(prediction_date, time_steps=10):
    """
    Predict future stock prices based on historical data up to the start_date.
    
    :param ticker: The stock ticker symbol (e.g., 'GLD' for gold ETF)
    :param start_date: The start date for downloading the historical data
    :param end_date: The end date for downloading the historical data
    :param prediction_date: The future date to predict the stock price for
    :param time_steps: The number of time steps used by the model (default: 10)
    :return: Predicted price for the given future date
    """
    
    # Step 1: Load the trained model and scaler
    model = load_model('model.keras')  # Replace with your model file
    scaler = joblib.load('X_scaler.pkl')  # Replace with your scaler file
    Yscaler = joblib.load('Y_scaler.pkl')  # Replace with your scaler file

    data = pd.read_csv('commodity_price_data.csv')
    # Step 3: Preprocess the data (same as during training)
        # Ensure 'Date' is in datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Get the latest date in the DataFrame
    latest_date = data['date'].max()

    print("Latest Date:", latest_date)

    data['MA_10'] = data['close_price'].rolling(window=10,  min_periods=1).mean()
    mean_value = data['close_price'].mean()
    data['Momentum'] = data['close_price'].fillna(mean_value)
    delta = data['close_price'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    clean_data = data.dropna()
    clean_data['Volatility'] = clean_data['close_price'].rolling(window=20,min_periods=1).std()
    clean_data = clean_data.dropna()
    clean_data['Upper_BB'] = clean_data['MA_10'] + (clean_data['Volatility'] * 2)
    clean_data['Lower_BB'] = clean_data['MA_10'] - (clean_data['Volatility'] * 2)
    clean_data['date'] = pd.to_datetime(clean_data['date'])
    clean_data['Month'] = clean_data['date'].dt.month
    clean_data['Day_of_Week'] = clean_data['date'].dt.dayofweek
    clean_data = clean_data.drop(columns=['symbol', 'date'])
    clean_data['MA_50'] = clean_data['close_price'].rolling(window=50, min_periods=1).mean()
    clean_data['MA_200'] = clean_data['close_price'].rolling(window=200,min_periods=1).mean()
    print(clean_data.shape)
    training_columns = ['volume', 'RSI', 'Volatility', 'Month', 'Day_of_Week']

    # Filter the prediction data to only include the training columns
    data_for_prediction_filtered = clean_data[training_columns]
    scaled_data = scaler.transform(data_for_prediction_filtered)  # Scale using the same scaler
    
    # Step 4: Prepare the data for prediction (reshaping it into time steps)
    X_new = []
    for i in range(len(scaled_data) - time_steps):
        X_new.append(scaled_data[i:i + time_steps])
    
    X_new = np.array(X_new)
    
    # Step 5: Make recursive predictions for future dates (for 21st, 22nd, ..., 25th Dec)
    # Start with the last time step (December 20 data)
    current_input = X_new[-1:]  # Last data from historical data
    
    predictions = []
    num_predictions = 5
    # Predict for the next 'num_predictions' days (or as required)
    for _ in range(num_predictions):
        # Predict the next value
        y_pred = model.predict(current_input)

        # Inverse transform to get the predicted price back to the original scale
        predicted_price = Yscaler.inverse_transform(y_pred)

        predictions.append(predicted_price[0][0])

        # Update the input for the next prediction
        new_feature_vector = np.zeros((1, 1, current_input.shape[2]))  # Creating a new feature vector for the next day
        new_feature_vector[0, 0, 0] = predicted_price[0][0]
        
        # Append the new feature to the input, shifting the data
        current_input = np.append(current_input[:, 1:, :], new_feature_vector, axis=1)

        # Increment the last date to simulate the next day
        latest_date = latest_date + pd.Timedelta(days=1)

    print(predictions)
    return str(predictions[-1])  # Return the last predicted value (for the next day)

  

    
    # # Make predictions for the next 5 days (21st, 22nd, 23rd, 24th, 25th Dec)
    # for _ in range((pd.to_datetime(prediction_date) - latest_date).days):
    #     y_pred = model.predict(current_input)
        
    #     # Inverse transform to get the predicted price back to original scale
    #     predicted_price = Yscaler.inverse_transform(y_pred)
        
    #     predictions.append(predicted_price[0][0])

    #     # Update input
    #     new_feature_vector = np.zeros((1, 1, current_input.shape[2]))
    #     new_feature_vector[0, 0, 0] = y_pred[0, 0]
    #     current_input = np.append(current_input[:, 1:, :], new_feature_vector, axis=1)

    #     print(predictions)

    # return str(predictions[-1])

