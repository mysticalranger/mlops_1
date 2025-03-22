
import unittest
import json
import sys
import os

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        response = self.app.get('/health')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', data)

    def test_predict_valid_input(self):
        test_data = {
            "notifications": 5,
            "times_opened": 10,
            "day_of_week": 3,
            "month": 6
        }
        response = self.app.post('/predict', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predicted_usage_minutes', data)
        self.assertIn('notification', data)

if __name__ == '__main__':
    unittest.main()