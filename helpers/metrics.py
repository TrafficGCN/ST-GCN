# Author 2023 Thomas Fink

import os
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(predictions_np, actual_data_np):

    # Root Mean Square Error
    rmse = math.sqrt(mean_squared_error(actual_data_np, predictions_np))
    # Mean Absolute Error
    mae = mean_absolute_error(actual_data_np, predictions_np)
    
    # Handle potential division by zero for R2 calculation
    denom_r2 = ((actual_data_np-actual_data_np.mean())**2).sum()
    r2 = 1-((actual_data_np-predictions_np)**2).sum()/denom_r2 if denom_r2 != 0 else 0  # Set to 0 or another suitable default value
    
    # Handle potential division by zero for Variance calculation
    denom_var = np.var(actual_data_np)
    variance = 1 - (np.var(predictions_np - actual_data_np) / denom_var) if denom_var != 0 else 0  # Set to 0 or another suitable default value
    
    # Handle potential division by zero for F_norm calculation
    denom_f = np.linalg.norm(actual_data_np)
    F_norm = np.linalg.norm(predictions_np - actual_data_np) / denom_f if denom_f != 0 else 0  # Set to 0 or another suitable default value

    return rmse, mae, r2, variance, 1 - F_norm


def visualize_metric(metric_list, metric_name, output_path='output'):

    #print(f"Generating evaluation {metric_name}...")
    # Visualization
    plt.figure()
    plt.plot(metric_list, label='Training ' + metric_name, color='#111111')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title('Training ' + metric_name)
    plt.legend()
    plt.savefig(f'{output_path}/metrics_{metric_name}.jpg', dpi=500, bbox_inches='tight', pad_inches=0.1, format='jpg')
    plt.close()
    
    # Save to CSV
    df = pd.DataFrame(metric_list, columns=[metric_name])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    csv_path = os.path.join(output_path, f'metrics_{metric_name}.csv')
    df.to_csv(csv_path, index=False)