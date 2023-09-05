# Author 2023 Thomas Fink

import torch
import torch.nn as nn

from statsmodels.tsa.arima.model import ARIMA

class ARIMAModule:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q
        self.models = []
        self.fitted_models = []

    def fit(self, data):
        # data should be a 2D numpy array with each row being a separate time series
        for series in data:
            model = ARIMA(series, order=(self.p, self.d, self.q))
            fitted_model = model.fit()
            self.models.append(model)
            self.fitted_models.append(fitted_model)

    def forecast(self, steps):
        forecasts = [model.forecast(steps) for model in self.fitted_models]
        return torch.tensor(forecasts)
    
class ARIMA_NN(nn.Module):
    def __init__(self, hidden_channels, p, d, q, num_predictions):
        super(ARIMA_NN, self).__init__()
        self.arima = ARIMAModule(p, d, q)
        
        # Adjusting the linear layer's input size
        self.linear = nn.Linear(2 * num_predictions, hidden_channels)
        
        self.output_layer = nn.Linear(hidden_channels, num_predictions)
        self.num_predictions = num_predictions


    def forward(self, x, edge_index, edge_weight):
        # Use ARIMA forecast
        arima_forecast = self.arima.forecast(self.num_predictions)
        
        # Ensure arima_forecast and x have the same dtype
        arima_forecast = arima_forecast.to(x.dtype)
        
        # Combine the last few timesteps from x with the ARIMA forecast.
        # This is to ensure that the input retains its original sequence length
        combined_input = torch.cat((x[:, -self.num_predictions:], arima_forecast), dim=1)
        
        x = torch.relu(self.linear(combined_input))
        x = self.output_layer(x)
        return x
