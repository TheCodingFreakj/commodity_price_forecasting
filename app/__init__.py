# app/__init__.py

from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from apscheduler.schedulers.background import BackgroundScheduler
from app.config import Config
import redis

# Initialize the app, db, and scheduler
app = Flask(__name__)
app.config.from_object(Config)


# Configure Redis
redis_client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
db = SQLAlchemy(app)
scheduler = BackgroundScheduler()

# Import the routes after app is created to avoid circular import
from .routes import main
from .services import store_data
from .automate_commodity_forecasting import automate_commodity_forecasting_sarimax
app.register_blueprint(main)

CORS(app)
# Schedule the job to run periodically
# scheduler.add_job(store_data, 'interval', minutes=1)
# scheduler.add_job(automate_commodity_forecasting_sarimax, 'interval', minutes=1,max_instances=1)

# Start the scheduler
scheduler.start()

def create_app():
    return app
