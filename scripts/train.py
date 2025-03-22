import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
import os
from datetime import datetime

# Set MLflow experiment name
mlflow.set_experiment("screen_time_prediction")

# Load preprocessed data
data = pd.read_csv('data/processed_data.csv')

# Load the saved scaler (optional, since data is already scaled)
scaler = joblib.load('models/scaler.pkl')

# Define features and target
X = data[['Notifications', 'Times Opened', 'DayOfWeek', 'Month', 'Notifications_x_TimesOpened']]
y = data['Usage (minutes)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'random_state': 42
}

# Start a new MLflow run
with mlflow.start_run(run_name=f"screen_time_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Log parameters
    mlflow.log_params(params)
    
    # Train the model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, test_predictions)
    mse = mean_squared_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
    
    # Log metrics
    mlflow.log_metric("train_mae", mean_absolute_error(y_train, train_predictions))
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_r2", r2)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Save the model to disk (regular save)
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = 'models/versions'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f'{model_dir}/model_{model_version}.pkl')
    
    # Also save as the current model
    joblib.dump(model, 'models/model.pkl')

    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')
    print(f"Training complete! Model saved to models/model.pkl and versioned at models/versions/model_{model_version}.pkl")
    print(f"Run ID: {mlflow.active_run().info.run_id}")