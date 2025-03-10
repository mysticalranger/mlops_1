import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load data
data = pd.read_csv('data/screentime_analysis.csv')

# check for missing values and duplicates
print(data.isnull().sum())
print(data.duplicated().sum())

# convert Date column to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# encode the categorical 'App' column using one-hot encoding
data = pd.get_dummies(data, columns=['App'], drop_first=True)

# scale numerical features using MinMaxScaler
scaler = MinMaxScaler()
data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

# feature engineering
data['Previous_Day_Usage'] = data['Usage (minutes)'].shift(1)
data['Notifications_x_TimesOpened'] = data['Notifications'] * data['Times Opened']

# Save preprocessed data
data.to_csv('data/processed_data.csv', index=False)

print("Preprocessing complete! Saved to data/processed_data.csv")