# app/routes.py
from datetime import datetime
from flask import Blueprint, jsonify, request
import pandas as pd

from .redis_helpers import get_predictions_from_cache, save_predictions_to_cache

from .automate_commodity_forecasting import  get_prediction_by_date
from .services import fetch_commodity_data, validate_prediction


main = Blueprint('main', __name__)

# Route to fetch commodity data
@main.route("/fetch_data", methods=["GET"])
def fetch_data():
    symbol = request.args.get("symbol", "GC=F")  # Default is Gold (GC=F)
    data = fetch_commodity_data(symbol)
    return jsonify(data.tail().to_dict())  # Return the last 5 rows of the data


@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the selected date from the request
        selected_date = request.json.get('date')
        print(f"Selected Date: {selected_date}")

        selected_date_str= datetime.strptime(selected_date, '%Y-%m-%d')
        # Get the prediction for the selected date
        predicted_price = get_prediction_by_date(selected_date)

        if predicted_price is not None:
            # If the selected date is in the future, return only predicted_price
            if selected_date_str > datetime.now():
                return jsonify({
                    'predicted_price': predicted_price,
                    'source': 'model'
                }), 200
            

            # If the selected date is in the past (or a trained date), validate prediction
            absolute_error = validate_prediction(selected_date_str, predicted_price)

            # Return the predicted value and the error metric for past dates
            return jsonify({
                'predicted_price': predicted_price,
                'absolute_error': absolute_error,
                'source': 'model'
            }), 200
        else:
            return jsonify({
                'predicted_price': 'Model will rebuild to give this result in future',
                'absolute_error': 'NA',
                'source': 'model'
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
