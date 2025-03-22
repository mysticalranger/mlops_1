import pandas as pd
import joblib
import os
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelMonitor:
    def __init__(self, model_path='models/model.pkl', data_path='data/processed_data.csv'):
        self.model_path = model_path
        self.data_path = data_path
        self.monitoring_file = 'monitoring/model_metrics.json'
        
        # Create monitoring directory if it doesn't exist
        os.makedirs('monitoring', exist_ok=True)
        
    def load_model_and_data(self):
        try:
            model = joblib.load(self.model_path)
            data = pd.read_csv(self.data_path)
            return model, data
        except Exception as e:
            print(f"Error loading model or data: {e}")
            return None, None
    
    def calculate_metrics(self):
        model, data = self.load_model_and_data()
        if model is None or data is None:
            return None
        
        # Define features and target
        X = data[['Notifications', 'Times Opened', 'DayOfWeek', 'Month', 'Notifications_x_TimesOpened']]
        y = data['Usage (minutes)']
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mae': mean_absolute_error(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'r2': r2_score(y, predictions),
            'data_size': len(data)
        }
        
        return metrics
    
    def save_metrics(self, metrics):
        # Load existing metrics if available
        if os.path.exists(self.monitoring_file):
            with open(self.monitoring_file, 'r') as f:
                try:
                    history = json.load(f)
                except:
                    history = []
        else:
            history = []
        
        # Add new metrics
        history.append(metrics)
        
        # Save updated metrics
        with open(self.monitoring_file, 'w') as f:
            json.dump(history, f, indent=4)
    
    def run_monitoring(self):
        metrics = self.calculate_metrics()
        if metrics:
            self.save_metrics(metrics)
            print(f"Model monitoring complete. Metrics: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
            return metrics
        return None

    def check_data_drift(self, reference_data_path='data/reference_data.csv'):
        """Check for data drift between reference and current data"""
        model, current_data = self.load_model_and_data()
        
        try:
            reference_data = pd.read_csv(reference_data_path)
        except Exception as e:
            print(f"Error loading reference data: {e}")
            # First run - save current data as reference
            current_data.to_csv(reference_data_path, index=False)
            return {"drift_detected": False, "message": "First run, setting current data as reference"}
        
        # Calculate basic statistics for numeric columns
        ref_stats = reference_data.describe()
        curr_stats = current_data.describe()
        
        # Check for significant changes in mean and std
        drift_results = {}
        for col in ['Notifications', 'Times Opened', 'Usage (minutes)']:
            if col in reference_data.columns and col in current_data.columns:
                mean_diff_pct = abs(ref_stats[col]['mean'] - curr_stats[col]['mean']) / ref_stats[col]['mean'] * 100
                std_diff_pct = abs(ref_stats[col]['std'] - curr_stats[col]['std']) / ref_stats[col]['std'] * 100
                
                drift_results[col] = {
                    'mean_diff_pct': mean_diff_pct,
                    'std_diff_pct': std_diff_pct,
                    'drift_detected': mean_diff_pct > 15 or std_diff_pct > 20  # Thresholds
                }
        
        drift_detected = any(col_result['drift_detected'] for col_result in drift_results.values())
        
        return {
            "drift_detected": drift_detected,
            "details": drift_results,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def trigger_retraining_if_needed(self):
        drift_results = self.check_data_drift()
        if drift_results["drift_detected"]:
            print(f"Data drift detected! Details: {drift_results['details']}")
            print("Triggering model retraining...")
            # Import here to avoid circular imports
            import subprocess
            subprocess.run(["python", "scripts/train.py"])
            return True
        else:
            print("No significant data drift detected. Model retraining not needed.")
            return False

if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.run_monitoring()


    