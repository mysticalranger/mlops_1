from flask import Flask, render_template
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Load monitoring data
    metrics = []
    if os.path.exists('monitoring/model_metrics.json'):
        with open('monitoring/model_metrics.json', 'r') as f:
            metrics = json.load(f)
    
    # Generate metrics plot
    plot_url = None
    if metrics:
        df = pd.DataFrame(metrics)
        
        # Create figure with two subplots (separate graphs for each metric)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot MAE
        ax1.plot(df['timestamp'], df['mae'], marker='o', color='blue', linewidth=2)
        ax1.set_title('Mean Absolute Error Over Time')
        ax1.set_ylabel('MAE')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(df['mae']):
            ax1.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        
        # Plot R²
        ax2.plot(df['timestamp'], df['r2'], marker='o', color='green', linewidth=2)
        ax2.set_title('R² Score Over Time')
        ax2.set_ylabel('R²')
        ax2.set_xlabel('Timestamp')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 1.0)  # R² is typically between 0 and 1
        
        # Add value labels
        for i, v in enumerate(df['r2']):
            ax2.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Load recent predictions if available
    predictions = None
    if os.path.exists('data/predictions.csv'):
        predictions = pd.read_csv('data/predictions.csv').tail(10).to_dict('records')
    
    return render_template('index.html', 
                           metrics=metrics, 
                           plot_url=plot_url,
                           predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5001)