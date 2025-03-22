# Screen Time Wellness App

A machine learning application that predicts app usage time and provides wellness recommendations.

## Project Overview
This project uses machine learning to analyze screen time patterns and make predictions about usage time based on factors like notifications and app opening frequency. It aims to help users maintain digital wellbeing by providing personalized recommendations.

## Features
- Predicts app usage time based on notification count and app opening frequency
- Provides wellness recommendations based on predicted screen time
- REST API for easy integration with other applications
- Containerized deployment with Docker

## Technology Stack
- Python 3.9
- Flask for API development
- scikit-learn for machine learning
- Prefect for workflow orchestration
- Docker for containerization

## Setup Instructions
1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the preprocessing script:
   ```
   python scripts/preprocess.py
   ```
5. Train the model:
   ```
   python scripts/train.py
   ```
6. Start the Flask API:
   ```
   python app.py
   ```

## API Documentation
### Health Check
- **URL**: `/health`
- **Method**: GET
- **Response**: Status and whether model is loaded

### Predict Usage
- **URL**: `/predict`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "notifications": 10,
    "times_opened": 20,
    "day_of_week": 3,
    "month": 6
  }
  ```
- **Response**:
  ```json
  {
    "predicted_usage_minutes": 45.8,
    "notification": "All good!"
  }
  ```

## Docker Deployment
```
docker build -t wellness-app .
docker run -p 5000:5000 wellness-app
```

## Testing
Run the test suite:
```
python -m unittest discover tests
```