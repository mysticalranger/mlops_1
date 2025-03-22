import functools
from flask import request, jsonify
import os

# A simple API key validation function
def require_api_key(view_function):
    @functools.wraps(view_function)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key == os.getenv('API_KEY', 'default_dev_key'):
            return view_function(*args, **kwargs)
        else:
            return {'error': 'Invalid or missing API key'}, 401
    return decorated_function