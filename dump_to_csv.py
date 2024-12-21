import pandas as pd
from sqlalchemy import create_engine

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

def dump_data_to_csv(df, file_name):
    """
    Dumps the DataFrame to a CSV file.
    """
    df.to_csv(file_name, index=False)
    print(f"Data dumped into {file_name}")

if __name__ == "__main__":
    # Load data from the database
    data = load_data_from_db()
    
    # Specify the file name for the CSV
    csv_file_name = "commodity_price_data.csv"
    
    # Dump the data to the CSV file
    dump_data_to_csv(data, csv_file_name)
