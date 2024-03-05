# Author 2023 Thomas Fink

import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_file):
    data_df = pd.read_csv(csv_file, header=None)
    return data_df

def load_adjacency_matrix(csv_file):
    adjacency_df = pd.read_csv(csv_file)
    return adjacency_df

def normalize_data(data_df):
    scaler = MinMaxScaler()
    speed_normalized = scaler.fit_transform(data_df)
    return torch.tensor(speed_normalized, dtype=torch.float), scaler

def load_geocoordinates(csv_file):
    geocoordinates_df = pd.read_csv(csv_file)
    geocoordinates_df = geocoordinates_df.set_index("detid")
    return geocoordinates_df
