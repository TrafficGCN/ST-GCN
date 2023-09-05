# Author 2023 Thomas Fink

import os
import torch
import torch.nn as nn

import helpers.metrics as metrics
import helpers.visualisation as visualisation
import helpers.data as data
import helpers.stats as stats
import helpers.output as output

from model_config import init_model


if __name__ == "__main__":

    OS_PATH = os.path.dirname(os.path.realpath('__file__'))

    # Load your traffic data and adjacency matrix from CSV files
    DATA_SET = "metr-la"

    # Create subfolder in output_path
    speed_csv_file = f"{OS_PATH}/data/{DATA_SET}/speed.csv"
    adjacency_csv_file = f"{OS_PATH}/data/{DATA_SET}/adj.csv"

    speed_df = data.load_speed_data(speed_csv_file)[1:] # ignore the first row
    speed_df = speed_df.iloc[:,1:] # ignore the first column

    adjacency_df = data.load_adjacency_matrix(adjacency_csv_file)
    sensor_ids = adjacency_df.iloc[:, 0].tolist()  # Get the sensor IDs from the first column

    adjacency_df = adjacency_df.iloc[:,1:] # ignore the first column

    # Normalize speed data
    speed_normalized, scaler = data.normalize_speed_data(speed_df)
    traffic_data = speed_normalized

    traffic_data = torch.transpose(traffic_data, 0, 1)

    # Preprocess adjacency matrix
    adjacency_matrix = torch.tensor(adjacency_df.values, dtype=torch.float)
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
    edge_weight = adjacency_matrix[edge_index[0], edge_index[1]]

    # Create output file structure
    output_path = output.create_output_directories(OS_PATH, DATA_SET, sensor_ids)

    num_epochs = 1500
    num_predictions = 288
    train_data, test_data = traffic_data[:, :-num_predictions], traffic_data[:, -num_predictions:]

    # Hyperparameters set layers and settings for the desired model in model_config.py
    model = init_model(
        model_type = "GCN_LSTM_BI_Multi_Attention",
        train_data = train_data,
        num_predictions = num_predictions,
    )

    # Define the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.027, weight_decay=0) # 0.025 lstm
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # For collecting across all sensors
    loss_list, rmse_list, mae_list, r2_list, variance_list, accuracy_list = [], [], [], [], [], []

    # For collecting metrics for each sensor
    sensor_loss_lists, sensor_rmse_lists, sensor_mae_lists, sensor_r2_lists, sensor_variance_lists, sensor_accuracy_lists = [], [], [], [], [], []

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
            
        rmse, mae, r2, variance, accuracy = metrics.evaluate(predictions.detach().numpy().T, test_data.detach().numpy().T)
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}, Variance: {variance:.4f}, Accuracy: {accuracy:.4f}")
        loss_list, rmse_list, mae_list, r2_list, variance_list, accuracy_list = [ loss_list + [loss.item()], rmse_list + [rmse], mae_list + [mae], r2_list + [r2],variance_list + [variance], accuracy_list + [accuracy]]

        # Collect Individual sensor metrics
        sensor_losses, sensor_rmses, sensor_maes, sensor_r2s, sensor_variances, sensor_accuracies = [], [], [], [], [], []
        for sensor_idx, sensor_id in enumerate(sensor_ids):
            # if predictions' second dimension doesn't align with sensor_ids, 
            # this could raise an IndexError, so be cautious
            sensor_pred = predictions[sensor_idx, :].detach().numpy()
            sensor_actual = test_data[sensor_idx, :].detach().numpy()

            sensor_rmse, sensor_mae, sensor_r2, sensor_variance, sensor_accuracy = metrics.evaluate(sensor_pred, sensor_actual)
            
            sensor_losses, sensor_rmses, sensor_maes, sensor_r2s, sensor_variances, sensor_accuracies = [sensor_losses + [loss.item()], sensor_rmses + [sensor_rmse], sensor_maes + [sensor_mae], sensor_r2s + [sensor_r2], sensor_variances + [sensor_variance], sensor_accuracies + [sensor_accuracy]]
        
        sensor_loss_lists, sensor_rmse_lists, sensor_mae_lists, sensor_r2_lists, sensor_variance_lists, sensor_accuracy_lists = [sensor_loss_lists + [sensor_losses], sensor_rmse_lists + [sensor_rmses], sensor_mae_lists + [sensor_maes], sensor_r2_lists + [sensor_r2s], sensor_variance_lists + [sensor_variances], sensor_accuracy_lists + [sensor_accuracies]]

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
    actual_data_np = scaler.inverse_transform(test_data.detach().numpy().T)
    predictions_np = scaler.inverse_transform(predictions.detach().numpy().T)

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
