import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

def compare_models(data_path='data/processed_data.csv', models_dir='models/versions'):
    # Load test data
    data = pd.read_csv(data_path)
    X = data[['Notifications', 'Times Opened', 'DayOfWeek', 'Month', 'Notifications_x_TimesOpened']]
    y = data['Usage (minutes)']
    
    # Find all model versions
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("No model versions found!")
        return
    
    # Compare models
    results = []
    for model_file in model_files:
        model = joblib.load(os.path.join(models_dir, model_file))
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        version = model_file.replace('model_', '').replace('.pkl', '')
        results.append({
            'version': version,
            'mae': mae,
            'r2': r2
        })
    
    # Convert to DataFrame for better display
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('version')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results_df['version'], results_df['mae'], marker='o')
    plt.title('MAE by Model Version')
    plt.xticks(rotation=45)
    plt.ylabel('Mean Absolute Error')
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['version'], results_df['r2'], marker='o', color='green')
    plt.title('R² by Model Version')
    plt.xticks(rotation=45)
    plt.ylabel('R² Score')
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png')
    
    print(results_df)
    print(f"Best model by MAE: {results_df.iloc[results_df['mae'].idxmin()]['version']}")
    print(f"Best model by R²: {results_df.iloc[results_df['r2'].idxmax()]['version']}")
    
    return results_df

if __name__ == "__main__":
    compare_models()