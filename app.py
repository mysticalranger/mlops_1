from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'model.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    notifications = data['notifications']
    times_opened = data['times_opened']
    day_of_week = data['day_of_week']
    month = data['month']
    notifications_x_times_opened = notifications * times_opened
    X = [[notifications, times_opened, day_of_week, month, notifications_x_times_opened]]
    prediction = model.predict(X)[0]
    notification = "Take a break!" if prediction > 60 else "All good!"
    return jsonify({
        "predicted_usage_minutes": round(prediction, 2),
        "notification": notification
    })

if __name__ == '__main__':
    app.run(debug=True)