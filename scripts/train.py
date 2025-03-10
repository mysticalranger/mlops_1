import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load preprocessed data
data = pd.read_csv('data/processed_data.csv')

# Load the saved scaler (optional, since data is already scaled)
scaler = joblib.load('models/scaler.pkl')

# Define features and target
# Using all numerical columns except 'Usage (minutes)' and 'Date'
X = data[['Notifications', 'Times Opened', 'DayOfWeek', 'Month', 'Notifications_x_TimesOpened']]
y = data['Usage (minutes)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# # Save the model
# joblib.dump(model, 'models/model.pkl')

# print("Training complete! Model saved to models/model.pkl")