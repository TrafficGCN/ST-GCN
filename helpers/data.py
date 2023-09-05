# Author 2023 Thomas Fink

import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_speed_data(csv_file):
    speed_df = pd.read_csv(csv_file, header=None)
    return speed_df

def load_adjacency_matrix(csv_file):
    adjacency_df = pd.read_csv(csv_file)
    return adjacency_df

def normalize_speed_data(speed_df):
    scaler = MinMaxScaler()
    speed_normalized = scaler.fit_transform(speed_df)
    return torch.tensor(speed_normalized, dtype=torch.float), scaler
