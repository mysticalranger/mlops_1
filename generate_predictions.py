import pandas as pd
import joblib
import os

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)

# Load model and data
model = joblib.load('models/model.pkl')
data = pd.read_csv('data/processed_data.csv')

# Define features
X = data[['Notifications', 'Times Opened', 'DayOfWeek', 'Month', 'Notifications_x_TimesOpened']]

# Make predictions
predictions = model.predict(X)

# Add predictions to dataset
data['Predicted_Usage'] = predictions
data['Notification'] = data['Predicted_Usage'].apply(lambda x: "Take a break!" if x > 60 else "All good!")

# Save predictions
data.to_csv('data/predictions.csv', index=False)
print("Predictions saved to data/predictions.csv")