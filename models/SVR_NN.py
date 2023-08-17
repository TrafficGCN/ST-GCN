# Author 2023 Thomas Fink

import torch
import torch.nn as nn
import numpy as np

from sklearn.svm import SVR

class SVRModule:
    def __init__(self, kernel='rbf', degree=3, C=1.0, epsilon=0.1):
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.epsilon = epsilon
        self.models = []

    def fit(self, data):
        # data should be a 2D numpy array with each row being a separate time series
        for series in data:
            model = SVR(kernel=self.kernel, degree=self.degree, C=self.C, epsilon=self.epsilon)
            model.fit(np.arange(len(series)).reshape(-1, 1), series)
            self.models.append(model)

    def forecast(self, steps):
        forecasts = [model.predict(np.arange(len(model.support_), len(model.support_) + steps).reshape(-1, 1)) for model in self.models]
        return torch.tensor(forecasts)

class SVR_NN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel='rbf', degree=3, C=1.0, epsilon=0.1, num_predictions=288):
        super(SVR_NN, self).__init__()
        self.svr = SVRModule(kernel, degree, C, epsilon)
        
        # Adjusting the linear layer's input size
        self.linear = nn.Linear(2 * num_predictions, hidden_size)
        
        self.output_layer = nn.Linear(hidden_size, num_predictions)
        self.num_predictions = num_predictions

    def forward(self, x, edge_index, edge_weight):
        # Use SVR forecast
        svr_forecast = self.svr.forecast(self.num_predictions)
        
        # Ensure svr_forecast and x have the same dtype
        svr_forecast = svr_forecast.to(x.dtype)
        
        # Combine the last few timesteps from x with the SVR forecast.
        combined_input = torch.cat((x[:, -self.num_predictions:], svr_forecast), dim=1)
        
        x = torch.relu(self.linear(combined_input))
        x = self.output_layer(x)
        return x
