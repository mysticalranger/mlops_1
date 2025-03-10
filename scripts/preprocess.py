import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load data
data = pd.read_csv('C:/Users/riyan/Downloads/project_1/wellness_app_project/screentime_analysis.csv')

# Check for missing values and duplicates
print("Missing values:\n", data.isnull().sum())
print("Exact duplicates:", data.duplicated().sum())

# Convert Date to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# Average duplicates by Date and App
data = data.groupby(['Date', 'App']).mean().reset_index()

# Convert Date again after groupby (it may revert to object)
data['Date'] = pd.to_datetime(data['Date'])

# Scale numerical features
scaler = MinMaxScaler()
data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

# Save the scaler for later use (e.g., in training/prediction)
joblib.dump(scaler, 'models/scaler.pkl')

# Feature engineering: Interaction term
data['Notifications_x_TimesOpened'] = data['Notifications'] * data['Times Opened']

# Save preprocessed data
data.to_csv('data/processed_data.csv', index=False)

print("Preprocessing complete! Saved to data/processed_data.csv")