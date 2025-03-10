from prefect import flow, task
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

@task
def preprocess_data():
    data = pd.read_csv('data/screentime_analysis.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data = data.groupby(['Date', 'App']).mean().reset_index()
    scaler = MinMaxScaler()
    data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])
    joblib.dump(scaler, 'models/scaler.pkl')
    data['Notifications_x_TimesOpened'] = data['Notifications'] * data['Times Opened']
    data.to_csv('data/processed_data.csv', index=False)
    return data

@task
def train_model(data):
    X = data[['Notifications', 'Times Opened', 'DayOfWeek', 'Month', 'Notifications_x_TimesOpened']]
    y = data['Usage (minutes)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/model.pkl')

@task
def predict_and_log():
    data = pd.read_csv('data/processed_data.csv')
    model = joblib.load('models/model.pkl')
    X = data[['Notifications', 'Times Opened', 'DayOfWeek', 'Month', 'Notifications_x_TimesOpened']]
    predictions = model.predict(X)
    data['Predicted_Usage'] = predictions
    data['Notification'] = data['Predicted_Usage'].apply(lambda x: "Take a break!" if x > 60 else "All good!")
    data.to_csv('data/predictions.csv', index=False)

@flow(name="screen_time_pipeline")
def screen_time_flow():
    data = preprocess_data()
    train_model(data)
    predict_and_log()

if __name__ == "__main__":
    screen_time_flow()