import json
from datetime import timedelta
from app import redis_client
CACHE_TTL = timedelta(hours=1)  # Time-to-live for cached predictions

def save_predictions_to_cache(selected_date, predictions):
    """
    Save predictions to Redis cache.
    """
    cache_key = f"predictions:{selected_date.strftime('%Y-%m-%d')}"
    redis_client.setex(cache_key, CACHE_TTL, json.dumps(predictions))

def get_predictions_from_cache(selected_date):
    """
    Retrieve predictions from Redis cache.
    """
    cache_key = f"predictions:{selected_date.strftime('%Y-%m-%d')}"
    cached_data = redis_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    return None
