
from datetime import datetime
import json
import os
import joblib
import numpy as np
import pandas as pd

from .redis_helpers import save_predictions_to_cache
from .services import  load_data_from_db, preprocess_data
from joblib import dump,load
from keras.models import load_model,save_model, Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarimax_model(X_train, y_train):
    """
    Fit a SARIMAX model with exogenous features and make predictions.
    """
    # print(f"y_train------------>{y_train}")
    # print(f"X_train------------>{X_train}")

    # Ensure y_train is numeric
    y_train = pd.to_numeric(y_train, errors='coerce')  # Convert to numeric and handle non-numeric values
    # print(f"y_train after conversion to numeric------------>{y_train}")

    # Ensure X_train only contains numeric values for the exogenous variables
    numeric_columns = ['open_price', 'high_price', 'low_price', 'volume']
    X_train_numeric = X_train[numeric_columns]
    # print(f"X_train_numeric------------>{X_train_numeric}")

    # Check if either X_train or y_train is empty before alignment
    if X_train_numeric.empty or y_train.empty:
        raise ValueError("Either X_train_numeric or y_train is empty before alignment!")

    # Handle any missing values (either drop or fill)
    X_train_numeric = X_train_numeric.ffill()
    y_train = y_train.ffill()
    # print(f"X_train_numeric after filling missing values------------>{X_train_numeric}")
    # print(f"y_train after filling missing values------------>{y_train}")

    # Check for duplicate indices in X_train_numeric and y_train
    if X_train_numeric.index.duplicated().any():
        print("Duplicate indices found in X_train_numeric, removing duplicates.")
        X_train_numeric = X_train_numeric[~X_train_numeric.index.duplicated()]

    if y_train.index.duplicated().any():
        print("Duplicate indices found in y_train, removing duplicates.")
        y_train = y_train[~y_train.index.duplicated()]

    # Ensure both X_train_numeric and y_train have the same datetime index
    if isinstance(X_train_numeric.index, pd.RangeIndex):  # Check if index is integer-based
        X_train_numeric.index = pd.date_range(start=y_train.index.min(), periods=len(X_train_numeric), freq='D')
    # print(f"X_train_numeric after adjusting datetime index------------>{X_train_numeric}")

    # Align the data by date (make sure both X_train_numeric and y_train have the same date index)
    X_train_numeric, y_train = X_train_numeric.align(y_train, join='inner', axis=0)
    # print(f"X_train_numeric after alignment------------>{X_train_numeric}")
    # print(f"y_train after alignment------------>{y_train}")

    # Check for empty DataFrames after alignment
    if X_train_numeric.empty or y_train.empty:
        raise ValueError("Training data (X_train_numeric or y_train) is empty after alignment!")

    # Ensure the index of y_train is datetime
    y_train.index = pd.to_datetime(y_train.index)
    y_train = y_train.asfreq('D')  # Set the frequency to daily
    # print(f"y_train after ensuring datetime and setting frequency------------>{y_train}")

    # Ensure the index of X_train_numeric is datetime and set the frequency
    X_train_numeric.index = pd.to_datetime(X_train_numeric.index)
    X_train_numeric = X_train_numeric.asfreq('D')  # Set the frequency to daily
    # print(f"X_train_numeric after ensuring datetime and setting frequency------------>{X_train_numeric}")

    # Ensure both are numeric and of type float64
    X_train_numeric = X_train_numeric.astype('float64')
    y_train = y_train.astype('float64')
    # print(f"X_train_numeric after conversion to float64------------>{X_train_numeric}")
    # print(f"y_train after conversion to float64------------>{y_train}")

    # SARIMAX model (X_train_numeric as exogenous variables, y_train as the target)
    model = SARIMAX(y_train, exog=X_train_numeric, order=(5, 1, 0))  # Adjust order as necessary
    print("SARIMAX model initialized.")

    # Try fitting with different methods
    try:
        model_fit = model.fit(method='BFGS')  # BFGS method with more iterations
        print("Model fitted successfully with BFGS.")
    except Exception as e:
        print(f"Error during fitting: {e}")
        model_fit = model.fit(method='Powell')
        print("Model fitted successfully with Powell.")

    return model_fit





def train_or_update_sarimax_model(X_train, y_train, model_path='models/sarimax_model.pkl'):
    """
    Load or train a new SARIMAX model with the provided training data.
    """
    # Check if the SARIMAX model already exists
    if os.path.exists(model_path):
        print("Updating SARIMAX model...")
        try:
            # Load the existing model
            sarimax_model = load(model_path)
            
            # Ensure it's a fitted SARIMAX model and update it
            if isinstance(sarimax_model, SARIMAX):
                print("Re-training SARIMAX model...")
                sarimax_model = fit_sarimax_model(X_train, y_train)
            else:
                print("Loaded model is not a SARIMAX model. Re-training from scratch.")
                sarimax_model = fit_sarimax_model(X_train, y_train)
        except Exception as e:
            print(f"Error loading SARIMAX model: {e}. Training a new SARIMAX model...")
            sarimax_model = fit_sarimax_model(X_train, y_train)
    else:
        print("Training new SARIMAX model with 1 year of data...")
        sarimax_model = fit_sarimax_model(X_train, y_train)

    # Save the trained or updated model
    dump(sarimax_model, model_path)
    return sarimax_model




def rolling_prediction_and_update(sarimax_model, X_train, y_train, X_test, test_data, numeric_columns=['open_price', 'high_price', 'low_price', 'volume'], future_steps=5):
    """
    Perform rolling predictions and update the SARIMAX model with new data.
    The prediction now includes both the predicted price and the corresponding date.
    """
    predictions = []
    X_test_selected = X_test[numeric_columns]  # Select only the relevant columns
    
    # Loop through the test set and forecast future values
    for i in range(len(X_test_selected)):
        # Forecast the next value using the most recent data
        forecast = sarimax_model.forecast(steps=1, exog=X_test_selected.iloc[i:i+1])  # Use X_test for exogenous data
        print(f"Forecast for {X_test_selected.index[i]}: {forecast}")
        
        # Extract the prediction (forecasted value) and date
        predicted_price = forecast[0]  # The predicted price value
        prediction_date = X_test_selected.index[i]  # The corresponding date for the prediction
        
        # Append the prediction (price, date)
        predictions.append({'date': prediction_date, 'predicted_price': predicted_price})
        
        # Update the model with the next data point (retrain using the extended data)
        new_data = test_data.iloc[i:i+1]['close_price']
        print(f"Updating model with new data: {new_data}")
        
        # Append new data to the training set
        y_train = pd.concat([y_train, new_data], axis=0)  # Append new data to the target (y_train)
        X_train = pd.concat([X_train, X_test.iloc[i:i+1]], ignore_index=True)  # Append new feature data to X_train
        
        # Re-train the SARIMAX model with the updated data after every step
        if i % 10 == 0:  # Retrain every 10 steps
            sarimax_model = fit_sarimax_model(X_train, y_train)

    # Make future predictions (beyond the test data)
    future_forecasts = sarimax_model.forecast(steps=future_steps, exog=X_test_selected.iloc[-future_steps:])  # Predict beyond the last available data
    
    # Store the future predictions
    last_date = X_test_selected.index[-1]  # Last date in the test set
    for step in range(future_steps):
        future_date = last_date + pd.Timedelta(days=step + 1)  # Predict the next days
        predictions.append({'date': future_date, 'predicted_price': future_forecasts[step]})
    
    return predictions



def save_predictions_to_cache(predictions, prediction_path='models/predictions.json'):
    """
    Store the predictions in a JSON file.
    """
    # Load existing predictions if the file exists
    if os.path.exists(prediction_path):
        with open(prediction_path, 'r') as file:
            stored_predictions = json.load(file)
    else:
        stored_predictions = []  # Initialize an empty list if the file does not exist

    # Format the predictions as a list of dictionaries with date and predicted price
    formatted_predictions = []
    for prediction in predictions:
        # Ensure the 'date' field is a pandas Timestamp, then convert to string
        if isinstance(prediction['date'], pd.Timestamp):
            prediction_date = prediction['date'].strftime('%Y-%m-%d')
        else:
            prediction_date = prediction['date']

        # Convert np.float64 to standard float
        if isinstance(prediction['predicted_price'], np.float64):
            predicted_price = float(prediction['predicted_price'])
        else:
            predicted_price = prediction['predicted_price']

        formatted_predictions.append({
            'date': prediction_date,
            'predicted_price': predicted_price
        })

    # Append the new predictions to the stored predictions
    stored_predictions.extend(formatted_predictions)

    # Save the updated predictions to file
    try:
        with open(prediction_path, 'w') as file:
            json.dump(stored_predictions, file, indent=4)
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def get_prediction_by_date(selected_date, prediction_path='models/predictions.json'):
    """
    Retrieve the prediction for a specific date from the cache if it exists.
    Does not create or update the cache, only checks and returns the result.
    """
    # Convert selected_date to string format (yyyy-mm-dd)
    # Convert selected_date string to a datetime object
    selected_date = datetime.strptime(selected_date, '%Y-%m-%d')

    # Convert selected_date to string format (yyyy-mm-dd)
    selected_date_str = selected_date.strftime('%Y-%m-%d')

    # Check if the prediction cache file exists
    if os.path.exists(prediction_path):
        # If the file exists, load the stored predictions
        with open(prediction_path, 'r') as file:
            stored_predictions = json.load(file)
        
        # Iterate over the stored predictions and check if any match the selected_date_str
        for prediction in stored_predictions:
            if prediction['date'] == selected_date_str:
                print(f"Prediction for {selected_date_str} found in cache.")
                return prediction['predicted_price']
        
        # If no matching date is found
        print(f"No prediction found for {selected_date_str} in the cache.")
        return None
    else:
        print(f"Prediction cache not found at {prediction_path}.")
        return None
    

def automate_commodity_forecasting_sarimax(selected_date=None):
    print("Starting model update...")

    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)

    # Load data from the database
    data = load_data_from_db()  # Assuming this returns a DataFrame
    
    # Preprocess the data (assuming it returns a time series)
    preprocessed_data = preprocess_data(data)  # This should return a time series or DataFrame
    preprocessed_data = preprocessed_data.sort_index()

    # Set the date index frequency to daily
    preprocessed_data.index = pd.to_datetime(preprocessed_data.index)
    preprocessed_data = preprocessed_data.asfreq('D')  # Assuming daily data

    # Handle the selected date
    if selected_date:
        if isinstance(selected_date, str):
            selected_date = datetime.strptime(selected_date, '%Y-%m-%d')
    else:
        selected_date = datetime.today()  # Default to today's date if not provided

    selected_date = pd.to_datetime(selected_date)

    # Filter data until the selected date
    filtered_data = preprocessed_data.loc[preprocessed_data.index <= selected_date]

    # Use the past 365 days of data for training and last 60 days for testing
    train_data = filtered_data[-365:]
    test_data = filtered_data[-60:]
    
    # Define X_train and y_train properly
    X_train, y_train = train_data.drop('close_price', axis=1), train_data['close_price']
    X_test, y_test = test_data.drop('close_price', axis=1), test_data['close_price']

    # Train or update the SARIMAX model
    sarimax_model = train_or_update_sarimax_model(X_train, y_train)

    # Perform rolling prediction and update the model with new data
    predictions = rolling_prediction_and_update(sarimax_model, X_train, y_train, X_test, test_data)

    save_predictions_to_cache(predictions)







# Function to fit an LSTM model
def fit_lstm_model(X_train, y_train):
        # Convert X_train to a NumPy array if it's a DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    # Reshape X_train if it's 2D (samples, features) to 3D (samples, time_steps, features)
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Add time_step dimension (1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    return model

# Function to save a model
def save_model(model, path):
    if model is not None and hasattr(model, 'save'):  # Check if the model is a Keras model
        model.save(path)
        print(f"Model saved at {path}.")
    else:
        print("Error: The object is not a valid Keras model.")



