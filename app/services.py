
# app/services.py
import joblib
import pandas as pd
import yfinance as yf
from .model import create_commodity_price_model
from . import db
from datetime import datetime
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler



# Create the model class with the db object
CommodityPrice = create_commodity_price_model(db)
# Fetch commodity data from Yahoo Finance
def fetch_commodity_data(symbol="GC=F"):
    data = yf.download(symbol, period="1y", interval="1d") # Download historical data (e.g., last 1 year of daily data)
    print(f"from function fetch_commodity_data------------>{data}")
    return data

def store_data():
    from . import app
    with app.app_context():  # Manually push the app context
        from . import db
        symbol = "GC=F"  # Example: Gold futures (you can change this as needed)
        data = fetch_commodity_data(symbol)

        print(f"fetch_commodity_data------------>{data}")

        if data.empty:
            print("No data returned from Yahoo Finance.")
            return  # Exit if no data is available

        # Convert the fetched data into database entries
        for index, row in data.iterrows():
            # Convert the index to a datetime object if it isn't already
            date = index.to_pydatetime() if isinstance(index, pd.Timestamp) else index

            # Check if the record already exists in the database
            existing_record = CommodityPrice.query.filter_by(symbol=symbol, date=date).first()

            if existing_record:
                print(f"Record for {symbol} on {date} already exists. Skipping.")
                continue  # Skip adding this record

            # Convert np.float64 to native Python float
            open_price = float(row['Open'])
            high_price = float(row['High'])
            low_price = float(row['Low'])
            close_price = float(row['Close'])
            volume = int(row['Volume'])

            # Create a CommodityPrice object and populate the fields
            commodity_price = CommodityPrice(
                symbol=symbol,
                date=date,  # Ensure date is a datetime object
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume
            )
            # Add commodity_price to the session (if using SQLAlchemy)
            db.session.add(commodity_price)

            # Optional: Print out to verify the data is correct
            print(f"Added new record: {commodity_price}")

        # Commit to save the data to the database
        db.session.commit()
        print("Data successfully stored in the database.")



def validate_prediction(selected_date_str, predicted_price):
    """
    Compares the predicted price to the actual price from the database.
    Returns the error metric (e.g., absolute error).
    """
    # Convert the string to datetime
    if isinstance(selected_date_str, datetime):
        selected_date = selected_date_str
    else:
        # Convert the string to datetime if it is a string
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')
    symbol = "GC=F"
    # Query the actual price from the database
   
    actual_price =  CommodityPrice.query.filter_by(symbol=symbol, date=selected_date).first()

    if actual_price is None:
        return None, "Actual price not available for this date and symbol."

    # Use the close price for comparison (you can change this based on the model's target)
    actual_price_value = actual_price.close_price
    print(actual_price_value)
    print(predicted_price)
    # Calculate the absolute error
    absolute_error = abs(predicted_price - actual_price_value)

    percentage_error = (absolute_error / actual_price_value) * 100

    return percentage_error, None

# 1. Function to load data from the database
def load_data_from_db():
    """
    Connects to the PostgreSQL database and loads commodity price data into a DataFrame.
    Adjust the database connection parameters accordingly.
    """
    # Example connection string; adjust for your database
    engine = create_engine('postgresql://avnadmin:AVNS_SPx5mGZsHfWLTBIzEGM@pg-39985733-pallavidapriya75-97f0.h.aivencloud.com:12783/defaultdb')
    query = "SELECT * FROM commodity_price;"  # Adjust your table name as needed
    df = pd.read_sql(query, engine)
    print("Data loaded from database")
    
    return df

def preprocess_data(df):
    """
    Preprocess the commodity price data.
    This can include things like parsing dates, filling missing values, and scaling numeric features.
    """
    # Ensure 'date' column is in datetime format
    if 'date' not in df.columns:
        raise ValueError("'date' column is missing from the DataFrame.")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Handle invalid date formats
    df.set_index('date', inplace=True)

    # Remove duplicates, keeping the last occurrence
    df = df.loc[~df.index.duplicated(keep='last')]

    # Resample the data to daily frequency and forward fill missing values
    df = df.asfreq('D')  # Set the frequency to daily ('D')
    
    # Fill missing values using forward fill (can be adjusted to other methods)
    df.ffill(inplace=True)

    # Sorting by date (this is mostly redundant since the index is already the date)
    df.sort_index(inplace=True)

    # Check for any remaining missing values after filling
    if df.isnull().values.any():
        print("Warning: There are still missing values in the DataFrame.")

    # Identify numeric columns to scale
    # numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    # if len(numeric_columns) > 0:
    #     # Apply StandardScaler to numeric columns
    #     scaler = StandardScaler()
    #     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    #     print("Numeric columns scaled using StandardScaler.")
    
    # Print a summary of the processed data (first few rows and info)
    print(f"Data preprocessing complete. Data summary:")
    print(df.head())  # Show the first few rows of the data
    print(f"Data types after preprocessing: \n{df.dtypes}")

    return df




