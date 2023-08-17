# Author 2023 Thomas Fink

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import pandas as pd

def save_predictions_to_csv(predictions, output_folder):
    # Create a DataFrame from the predictions array
    df = pd.DataFrame(predictions)

    # Set the column names
    column_names = ['timestep ' + str(i + 1) for i in range(predictions.shape[1])]
    df.columns = column_names

    # Add the sensor names as a new column
    df.insert(0, 'sensor', ['sensor ' + str(i + 1) for i in range(predictions.shape[0])])

    # Save the DataFrame to a CSV file
    df.to_csv(f'{output_folder}/predictions.csv', index=False)



def plot_predictions(predictions, actual, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (pred, actuals) in enumerate(zip(predictions, actual)):
        plt.figure()
        plt.plot(pred, label='Predicted', color='#FFDC00', linewidth=0.76)
        plt.plot(actuals, label='Actual', color='#111111', linewidth=0.75)
        plt.xlabel('Time Step')
        plt.ylabel('Speed')
        plt.title(f'Sensor {i} - Predictions vs Actual')
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
        plt.savefig(f'{output_folder}/sensor_{i}_predictions.jpg', dpi=500, bbox_inches='tight', pad_inches=0.1, format='jpg')
        plt.close()



def visualize_predictions(predictions, actual, output_folder='output'):
    plot_predictions(predictions, actual, output_folder)
    save_predictions_to_csv(predictions, output_folder)
    

def visualize_metric(metric_list, metric_name, output_folder='output'):
    plt.figure()
    plt.plot(metric_list, label='Training ' + metric_name, color='#111111')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title('Training ' + metric_name)
    plt.legend()
    plt.savefig(f'{output_folder}/metrics_{metric_name}.jpg', dpi=500, bbox_inches='tight', pad_inches=0.1, format='jpg')
    plt.close()
