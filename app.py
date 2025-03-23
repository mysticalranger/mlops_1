from flask import Flask, request, jsonify, render_template
import joblib
import os
import logging
from dotenv import load_dotenv
from flask_restx import Api, Resource, fields
import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import base64

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'))
api = Api(app, version='1.0', title='Screen Time Wellness API',
          description='A machine learning API for screen time predictions and wellness recommendations',
          doc='/api/docs',
          prefix='/api')  # Add this line to restrict API routes to /api

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
ns = api.namespace('', description='Wellness predictions')

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
            if not (0 <= notifications <= 100):
                return {"error": "notifications must be between 0 and 100"}, 400
            if not (0 <= times_opened <= 50):
                return {"error": "times_opened must be between 0 and 50"}, 400
            
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

@ns.route('/model-range')
class ModelRange(Resource):
    @api.doc(description='Get the model prediction range (min, max, avg)')
    @api.response(200, 'Success')
    @api.response(503, 'Model Not Loaded', error_model)
    def get(self):
        """Get model prediction range"""
        if model is None:
            return {"error": "Model not loaded. Please train the model first."}, 503
        
        try:
            range_data = get_model_prediction_range()
            return range_data
        
        except Exception as e:
            logger.error(f"Error getting model range: {str(e)}")
            return {"error": "Failed to get model range"}, 500

def get_model_prediction_range():
    """Calculate the min and max possible predictions from the model"""
    try:
        # Load data to understand feature ranges
        data = pd.read_csv('data/processed_data.csv')
        
        # Make predictions with the model
        X = data[['Notifications', 'Times Opened', 'DayOfWeek', 'Month', 'Notifications_x_TimesOpened']]
        predictions = model.predict(X)
        
        return {
            "min": float(round(predictions.min(), 2)),
            "max": float(round(predictions.max(), 2)),
            "avg": float(round(predictions.mean(), 2))
        }
    except Exception as e:
        logger.error(f"Error calculating prediction range: {str(e)}")
        return {"min": 0, "max": 0, "avg": 0}

# ======= DASHBOARD ROUTES (Integrated from dashboard.py) =======


@app.route('/')
def dashboard():
    # Load monitoring data
    metrics = []
    if os.path.exists('monitoring/model_metrics.json'):
        with open('monitoring/model_metrics.json', 'r') as f:
            metrics = json.load(f)
    
    # Generate metrics plot
    plot_url = None
    if metrics:
        df = pd.DataFrame(metrics)
        
        # Create figure with two subplots (separate graphs for each metric)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot MAE
        ax1.plot(df['timestamp'], df['mae'], marker='o', color='blue', linewidth=2)
        ax1.set_title('Mean Absolute Error Over Time')
        ax1.set_ylabel('MAE')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(df['mae']):
            ax1.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        
        # Plot R²
        ax2.plot(df['timestamp'], df['r2'], marker='o', color='green', linewidth=2)
        ax2.set_title('R² Score Over Time')
        ax2.set_ylabel('R²')
        ax2.set_xlabel('Timestamp')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 1.0)  # R² is typically between 0 and 1
        
        # Add value labels
        for i, v in enumerate(df['r2']):
            ax2.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Load recent predictions if available
    predictions = None
    if os.path.exists('data/predictions.csv'):
        predictions = pd.read_csv('data/predictions.csv').tail(10).to_dict('records')
    
    return render_template('index.html', 
                          metrics=metrics, 
                          plot_url=plot_url,
                          predictions=predictions)

# Add a menu route to navigate between dashboard and API docs
@app.route('/menu')
def menu():
    return """
    <html>
    <head>
        <title>Screen Time Wellness - Menu</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #333; }
            .container { max-width: 800px; margin: 0 auto; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
            .btn { display: inline-block; background: #4CAF50; color: white; padding: 10px 20px; 
                  text-decoration: none; border-radius: 4px; margin-right: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Screen Time Wellness App</h1>
            
            <div class="card">
                <h2>Navigation Menu</h2>
                <p>Welcome to the Screen Time Wellness application. Choose an option below:</p>
                
                <a href="/" class="btn">Dashboard</a>
                <a href="/api/docs" class="btn">API Documentation</a>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)