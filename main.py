# Author 2023 Thomas Fink

import math
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn

import visualization
import data_helpers
import numpy as np

def evaluate(predictions_np, actual_data_np):

    # Root Mean Square Error
    rmse = math.sqrt(mean_squared_error(actual_data_np, predictions_np))
    # Mean Absolute Error
    mae = mean_absolute_error(actual_data_np, predictions_np)
    r2 = 1-((actual_data_np-predictions_np)**2).sum()/((actual_data_np-actual_data_np.mean())**2).sum()
    #r2 = metrics.r2_score(actual_data_np, predictions_np)
    variance = 1 - (np.var(predictions_np - actual_data_np) / np.var(actual_data_np))
    F_norm = np.linalg.norm(predictions_np - actual_data_np) / np.linalg.norm(actual_data_np)
    return rmse, mae, r2, variance, 1 - F_norm


if __name__ == "__main__":

    OS_PATH = os.path.dirname(os.path.realpath('__file__'))

    # Load your traffic data and adjacency matrix from CSV files
    speed_csv_file = OS_PATH + '/data/metr-la/metr_la_speed.csv'
    adjacency_csv_file = OS_PATH + '/data/metr-la/metr_la_adj.csv'

    speed_df = data_helpers.load_speed_data(speed_csv_file)[1:] # ignore the first row
    speed_df = speed_df.iloc[:,1:] # ignore the first column

    adjacency_df = data_helpers.load_adjacency_matrix(adjacency_csv_file)[1:] # ignore the first row
    adjacency_df = adjacency_df.iloc[:,1:] # ignore the first column


    # Normalize speed data
    speed_normalized, scaler = data_helpers.normalize_speed_data(speed_df)
    traffic_data = speed_normalized

    traffic_data = torch.transpose(traffic_data, 0, 1)

    # Preprocess adjacency matrix
    adjacency_matrix = torch.tensor(adjacency_df.values, dtype=torch.float)
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
    edge_weight = adjacency_matrix[edge_index[0], edge_index[1]]

    # Hyperparameters
    num_gcn_layers = 16  # Reduce the number of layers 16
    num_rnn_layers = 3 #3
    hidden_channels = 32 #32
    num_predictions = 288  # Number of timesteps to predict into the future #288
    dropout = 0 #0
    num_epochs = 1500


    # Assuming traffic_data is your entire dataset
    train_data, test_data = traffic_data[:, :-num_predictions], traffic_data[:, -num_predictions:]

    # Create the model
    model_type = 'GCN_LSTM_BI'  # change this to the desired model type

    if model_type == 'ARIMA_NN':
        from models.ARIMA_NN import ARIMA_NN
        p, d, q = 5, 1, 0  # Sample ARIMA parameters
        hidden_channels = 64
        model = ARIMA_NN(train_data.size(1), hidden_channels, p, d, q, num_predictions)
        train_data = train_data.to(dtype=torch.float32)
        numpy_train_data = train_data.numpy()
        model.arima.fit(numpy_train_data)

    elif model_type == 'SVR':
        from models.SVR_NN import SVR_NN
        # Adjust the parameters as needed
        hidden_channels = 1
        kernel = 'rbf'
        degree = 3
        C = 1.0
        epsilon = 0.1

        model = SVR_NN(train_data.size(1), hidden_channels, kernel, degree, C, epsilon, num_predictions)
        train_data = train_data.to(dtype=torch.float32)
        numpy_train_data = train_data.numpy()
        model.svr.fit(numpy_train_data)

    elif model_type == 'GCN_LSTM':
        from models.GCN_LSTM import GCN_LSTM
        model = GCN_LSTM(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)

    elif model_type == 'GCN_GRU':
        from models.GCN_GRU import GCN_GRU
        model = GCN_GRU(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)

    elif model_type == 'GCN_LSTM_BI':
        from models.GCN_LSTM_BI import GCN_LSTM_BI
        model = GCN_LSTM_BI(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)

    elif model_type == 'GCN_LSTM_BI_Attention':
        from models.GCN_LSTM_BI_Attention import GCN_LSTM_BI_Attention
        model = GCN_LSTM_BI_Attention(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)

    elif model_type == 'GCN_LSTM_Peepholes':
        from models.GCN_LSTM_Peepholes import GCN_LSTM_Peepholes
        model = GCN_LSTM_Peepholes(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)

    elif model_type == 'GCN_GRU_BI':
        from models.GCN_GRU_BI import GCN_GRU_BI
        model = GCN_GRU_BI(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)

    elif model_type == 'GCN_Transformer':
        from models.GCN_Transformer import GCN_Transformer
        model = GCN_Transformer(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)

    elif model_type == 'GCN_LSTM_TeacherForcing':
        from models.GCN_LSTM_TeacherForcing import GCN_LSTM_TeacherForcing
        model = GCN_LSTM_TeacherForcing(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)

    elif model_type == 'GCN_LSTM_BI_TeacherForcing':
        from models.GCN_LSTM_BI_TeacherForcing import GCN_LSTM_BI_TeacherForcing
        model = GCN_LSTM_BI_TeacherForcing(train_data.size(1), hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)


    # Define the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.027, weight_decay=0) # 0.025 lstm
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    loss_list = []
    loss_list = []
    rmse_list = []
    mae_list = []
    r2_list = []
    variance_list = []
    accuracy_list = []


    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(train_data, edge_index, edge_weight)
        loss = criterion(predictions, test_data)
        loss.backward()
        optimizer.step()
        #scheduler.step()  

        # Set any negative predictions to zero in the final epoch
        if epoch == num_epochs - 1:
            predictions = torch.where(predictions < 0, torch.zeros_like(predictions), predictions)
            

        rmse, mae, r2, variance, accuracy = evaluate(predictions.detach().numpy().T, test_data.detach().numpy().T)
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}, Variance: {variance:.4f}, Accuracy: {accuracy:.4f}")
        loss_list.append(loss.item())
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        variance_list.append(variance)
        accuracy_list.append(accuracy)


    output_path = OS_PATH + "/output/metr-la/"


    # Visualize the Metrics
    visualization.visualize_metric(loss_list, 'Loss', output_path)
    visualization.visualize_metric(rmse_list, 'RMSE', output_path)
    visualization.visualize_metric(mae_list, 'MAE', output_path)
    visualization.visualize_metric(r2_list, 'R2 Score', output_path)
    visualization.visualize_metric(variance_list, 'Variance', output_path)
    visualization.visualize_metric(accuracy_list, 'Accuracy', output_path)


    # Convert the normalized predictions back to the original speed values
    actual_data_np = scaler.inverse_transform(test_data.detach().numpy().T)
    predictions_np = scaler.inverse_transform(predictions.detach().numpy().T)

    # You can transpose again if needed
    actual_data_np = actual_data_np.T
    predictions_np = predictions_np.T


    # Visualize the predictions
    visualization.visualize_predictions(predictions_np, actual_data_np, output_path)