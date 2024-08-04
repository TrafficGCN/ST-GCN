import os
from datetime import datetime
import torch
import torch.nn as nn
import time

import helpers.metrics as metrics
import helpers.visualisation as visualisation
import helpers.data as data
import helpers.stats as stats
import helpers.output as output

from model_config import init_model

def ensure_tensor_on_device(tensor, device):
    if device.type == 'mps':
        torch.mps.empty_cache()
    try:
        return tensor.to(device)
    except RuntimeError as e:
        print(f"Operation not supported on {device}: {e}, falling back to CPU")
        return tensor.to("cpu")

if __name__ == "__main__":
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))

    # Load your traffic data and adjacency matrix from CSV files
    DATA_SET = "metr-la"
    data_csv_file = f"{OS_PATH}/data/{DATA_SET}/traffic/speed.csv"
    adjacency_csv_file = f"{OS_PATH}/data/{DATA_SET}/traffic/adj.csv"

    data_df = data.load_data(data_csv_file)[1:]  # ignore the first row
    data_df = data_df.iloc[:, 1:]  # ignore the first column
    adjacency_df = data.load_adjacency_matrix(adjacency_csv_file)
    sensor_ids = adjacency_df.iloc[:, 0].tolist()  # Get the sensor IDs from the first column
    adjacency_df = adjacency_df.iloc[:, 1:]  # ignore the first column

    # Normalize speed data
    data_normalized, scaler = data.normalize_data(data_df)
    traffic_data = data_normalized

    # Move data to the correct device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    traffic_data = ensure_tensor_on_device(torch.transpose(traffic_data, 0, 1), device)
    adjacency_matrix = ensure_tensor_on_device(torch.tensor(adjacency_df.values, dtype=torch.float), device)

    # Handle potential CPU fallback for specific operations
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous().to("cpu")
    edge_weight = adjacency_matrix[edge_index[0], edge_index[1]].to("cpu")

    # Move edge_index and edge_weight back to the original device if it is not CPU
    if device != torch.device("cpu"):
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

    # Create output file structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_dataset = f"{timestamp}_{DATA_SET}"
    output_path = output.create_output_directories(OS_PATH, timestamped_dataset, sensor_ids)

    num_epochs = 1000
    num_predictions = 288
    train_data, test_data = traffic_data[:, :-num_predictions], traffic_data[:, -num_predictions:]

    # Hyperparameters set layers and settings for the desired model in model_config.py
    model = init_model(
        model_type="GCN_LSTM_BI",
        train_data=train_data,
        num_predictions=num_predictions,
    ).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # Define the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.027, weight_decay=0)

    # For collecting across all sensors
    loss_list, rmse_list, mae_list, r2_list, variance_list, accuracy_list = [], [], [], [], [], []

    # For collecting metrics for each sensor
    sensor_loss_lists, sensor_rmse_lists, sensor_mae_lists, sensor_r2_lists, sensor_variance_lists, sensor_accuracy_lists = [], [], [], [], [], []

    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(train_data, edge_index, edge_weight)
        loss = criterion(predictions, test_data)
        loss.backward()
        optimizer.step()
        
        # Clear cache after each batch if using MPS
        if device.type == 'mps':
            torch.mps.empty_cache()

        # Set any negative predictions to zero in the final epoch
        if epoch == num_epochs - 1:
            predictions = torch.where(predictions < 0, torch.zeros_like(predictions), predictions)

        rmse, mae, r2, variance, accuracy = metrics.evaluate(predictions.cpu().detach().numpy().T, test_data.cpu().detach().numpy().T)
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}, Variance: {variance:.4f}, Accuracy: {accuracy:.4f}")
        loss_list.append(loss.item())
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        variance_list.append(variance)
        accuracy_list.append(accuracy)

        # Collect Individual sensor metrics
        sensor_losses, sensor_rmses, sensor_maes, sensor_r2s, sensor_variances, sensor_accuracies = [], [], [], [], [], []
        for sensor_idx, sensor_id in enumerate(sensor_ids):
            sensor_pred = predictions[sensor_idx, :].cpu().detach().numpy()
            sensor_actual = test_data[sensor_idx, :].cpu().detach().numpy()

            sensor_rmse, sensor_mae, sensor_r2, sensor_variance, sensor_accuracy = metrics.evaluate(sensor_pred, sensor_actual)

            sensor_losses.append(loss.item())
            sensor_rmses.append(sensor_rmse)
            sensor_maes.append(sensor_mae)
            sensor_r2s.append(sensor_r2)
            sensor_variances.append(sensor_variance)
            sensor_accuracies.append(sensor_accuracy)

        sensor_loss_lists.append(sensor_losses)
        sensor_rmse_lists.append(sensor_rmses)
        sensor_mae_lists.append(sensor_maes)
        sensor_r2_lists.append(sensor_r2s)
        sensor_variance_lists.append(sensor_variances)
        sensor_accuracy_lists.append(sensor_accuracies)

    end_time = time.time()
    print(f"Training time on {device}: {end_time - start_time} seconds")

    # Visualize the Metrics
    metrics.visualize_metric(loss_list, 'Loss', output_path)
    metrics.visualize_metric(rmse_list, 'RMSE', output_path)
    metrics.visualize_metric(mae_list, 'MAE', output_path)
    metrics.visualize_metric(r2_list, 'R² Score', output_path)
    metrics.visualize_metric(variance_list, 'Variance', output_path)
    metrics.visualize_metric(accuracy_list, 'Accuracy', output_path)

    # Initialize lists for best values
    best_sensor_loss, best_sensor_rmse, best_sensor_mae = [float('inf')] * len(sensor_ids), [float('inf')] * len(sensor_ids), [float('inf')] * len(sensor_ids)
    best_sensor_r2, best_sensor_variance, best_sensor_accuracy = [float('-inf')] * len(sensor_ids), [float('-inf')] * len(sensor_ids), [float('-inf')] * len(sensor_ids)

    # Iterate through each sensor
    for sensor_idx in range(len(sensor_ids)):
        # Iterate through each epoch
        for epoch in range(num_epochs):
            # Update the best values
            best_sensor_loss[sensor_idx] = min(best_sensor_loss[sensor_idx], sensor_loss_lists[epoch][sensor_idx])
            best_sensor_rmse[sensor_idx] = min(best_sensor_rmse[sensor_idx], sensor_rmse_lists[epoch][sensor_idx])
            best_sensor_mae[sensor_idx] = min(best_sensor_mae[sensor_idx], sensor_mae_lists[epoch][sensor_idx])
            best_sensor_r2[sensor_idx] = max(best_sensor_r2[sensor_idx], sensor_r2_lists[epoch][sensor_idx])
            best_sensor_variance[sensor_idx] = max(best_sensor_variance[sensor_idx], sensor_variance_lists[epoch][sensor_idx])
            best_sensor_accuracy[sensor_idx] = max(best_sensor_accuracy[sensor_idx], sensor_accuracy_lists[epoch][sensor_idx])

    # Visualize the distributions
    stats.plot_error_distributions(best_sensor_rmse, best_sensor_mae, best_sensor_accuracy, best_sensor_r2, best_sensor_variance, output_path)

    # Load geocoordinates data
    geocoordinates_csv_file = f"{OS_PATH}/data/{DATA_SET}/geocoordinates.csv"

    if os.path.exists(geocoordinates_csv_file):
        geocoordinates_df = data.load_geocoordinates(geocoordinates_csv_file)
        stats.plot_error_distributions_map(best_sensor_rmse, best_sensor_mae, best_sensor_accuracy, best_sensor_r2, best_sensor_variance, output_path, sensor_ids, geocoordinates_df)
    else:
        print(f"File {geocoordinates_csv_file} does not exist! No heat maps will be created!")

    # Convert the normalized predictions back to the original speed values
    actual_data_np = scaler.inverse_transform(test_data.cpu().detach().numpy().T)
    predictions_np = scaler.inverse_transform(predictions.cpu().detach().numpy().T)

    # You can transpose again if needed
    actual_data_np = actual_data_np.T
    predictions_np = predictions_np.T

    # Visualize the predictions
    print("Saving predictions...")
    visualisation.save_predictions_to_csv(predictions_np, output_path)

    print("Generating sensor predictions...")
    for sensor_idx, (sensor_id, (pred, actual)) in enumerate(zip(sensor_ids, zip(predictions_np, actual_data_np))):
        sensor_id_str = str(int(sensor_id)) if isinstance(sensor_id, float) and sensor_id.is_integer() else str(sensor_id)
        sensor_folder = os.path.join(output_path, "sensors", f"sensor_{sensor_id_str}")

        visualisation.plot_prediction(pred, actual, sensor_id_str, sensor_folder)

    # Visualize the Metrics for each sensor
    print("Generating sensor metrics...")
    for sensor_idx, sensor_id in enumerate(sensor_ids):
        sensor_id_str = str(int(sensor_id)) if isinstance(sensor_id, float) and sensor_id.is_integer() else str(sensor_id)
        sensor_folder = os.path.join(output_path, "sensors", f"sensor_{sensor_id_str}")

        metrics.visualize_metric([x[sensor_idx] for x in sensor_loss_lists], 'Loss', sensor_folder)
        metrics.visualize_metric([x[sensor_idx] for x in sensor_rmse_lists], 'RMSE', sensor_folder)
        metrics.visualize_metric([x[sensor_idx] for x in sensor_mae_lists], 'MAE', sensor_folder)
        metrics.visualize_metric([x[sensor_idx] for x in sensor_r2_lists], 'R2 Score', sensor_folder)
        metrics.visualize_metric([x[sensor_idx] for x in sensor_variance_lists], 'Variance', sensor_folder)
        metrics.visualize_metric([x[sensor_idx] for x in sensor_accuracy_lists], 'Accuracy', sensor_folder)

if device.type == 'mps':
    torch.mps.empty_cache()
