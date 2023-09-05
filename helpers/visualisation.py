# Author 2023 Thomas Fink

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import pandas as pd

def save_predictions_to_csv(predictions, output_path):
    df = pd.DataFrame(predictions)
    column_names = ['timestep ' + str(i + 1) for i in range(predictions.shape[1])]
    df.columns = column_names
    df.insert(0, 'sensor', ['sensor ' + str(i + 1) for i in range(predictions.shape[0])])
    df.to_csv(f'{output_path}/predictions.csv', index=False)

def plot_prediction(pred, actual, sensor_id_str, output_path):

    plt.figure()
    plt.plot(pred, label='Predicted', color='#FFDC00', linewidth=0.76)
    plt.plot(actual, label='Actual', color='#111111', linewidth=0.75)
    plt.xlabel('Time Step')
    plt.ylabel('Speed')
    
    plt.title(f'Sensor {sensor_id_str} - Predictions vs Actual')
    
    num_timesteps = pred.shape[0]
    x_labels = []
    for t in range(num_timesteps):
        minutes = t * 5
        days = minutes // 1440
        hours = (minutes - days * 1440) // 60
        minutes = minutes % 60
        x_labels.append(f'{days:02d}:{hours:02d}:{minutes:02d}')
    plt.xticks(range(0, num_timesteps, 64), x_labels[::64])

    plt.legend()
    
    plt.savefig(f'{output_path}/sensor_{sensor_id_str}_predictions.jpg', dpi=500, bbox_inches='tight', pad_inches=0.1, format='jpg')
    
    plt.close()