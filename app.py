from flask import Flask, request, jsonify
import joblib
import os
import logging
from dotenv import load_dotenv
from flask_restx import Api, Resource, fields

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app, version='1.0', title='Screen Time Wellness API',
          description='A machine learning API for screen time predictions and wellness recommendations')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create models directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

# Load model with error handling
try:
    model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    else:
        logger.error(f"Model file not found at {model_path}")
        model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Define namespaces
ns = api.namespace('api', description='Wellness predictions')

# Define models for request/response
prediction_input = api.model('PredictionInput', {
    'notifications': fields.Float(required=True, description='Number of notifications received'),
    'times_opened': fields.Float(required=True, description='Number of times the app was opened'),
    'day_of_week': fields.Integer(required=True, description='Day of week (0-6, where 0 is Monday)'),
    'month': fields.Integer(required=True, description='Month (1-12)')
})

prediction_output = api.model('PredictionOutput', {
    'predicted_usage_minutes': fields.Float(description='Predicted usage time in minutes'),
    'notification': fields.String(description='Wellness recommendation')
})

error_model = api.model('ErrorModel', {
    'error': fields.String(description='Error message')
})

@ns.route('/health')
class Health(Resource):
    @api.doc(description='Check if the API is running and model is loaded')
    @api.response(200, 'Success')
    @api.response(503, 'Service Unavailable', error_model)
    def get(self):
        """Health check endpoint"""
        if model is not None:
            return {"status": "healthy", "model_loaded": True}
        else:
            return {"status": "unhealthy", "model_loaded": False}, 503

@ns.route('/predict')
class Predict(Resource):
    @api.doc(description='Make a prediction based on input data')
    @api.expect(prediction_input)
    @api.response(200, 'Success', prediction_output)
    @api.response(400, 'Validation Error', error_model)
    @api.response(503, 'Model Not Loaded', error_model)
    def post(self):
        """Prediction endpoint"""
        if model is None:
            return {"error": "Model not loaded. Please train the model first."}, 503
        
        try:
            # Validate request data
            data = request.json
            if not data:
                return {"error": "No data provided"}, 400
            
            # Extract and validate features
            notifications = float(data['notifications'])
            times_opened = float(data['times_opened'])
            day_of_week = int(data['day_of_week'])
            month = int(data['month'])
            
            if not (0 <= day_of_week <= 6):
                return {"error": "day_of_week must be between 0 and 6"}, 400
            if not (1 <= month <= 12):
                return {"error": "month must be between 1 and 12"}, 400
            
            # Make prediction
            notifications_x_times_opened = notifications * times_opened
            X = [[notifications, times_opened, day_of_week, month, notifications_x_times_opened]]
            prediction = model.predict(X)[0]
            threshold = float(os.getenv('MODEL_THRESHOLD', 60))
            notification = "Take a break!" if prediction > threshold else "All good!"
            
            logger.info(f"Prediction successful: {prediction} minutes")
            return {
                "predicted_usage_minutes": round(prediction, 2),
                "notification": notification
            }
        
        except ValueError as e:
            logger.error(f"Invalid input data: {str(e)}")
            return {"error": f"Invalid input data: {str(e)}"}, 400
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": "Failed to make prediction"}, 500

if __name__ == '__main__':
    app.run(debug=True)