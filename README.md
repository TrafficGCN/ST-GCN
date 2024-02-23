# Spatio-Temporal Graph Convolutional Network (ST-GCN)
Base models for graph convolutional networks.

### Citations

1. For the Los Angeles metr-la and Santa Clara pems-bay datasets cite: Kwak, Semin. (2020). PEMS-BAY and METR-LA in csv [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5146275

2. Weather sensors are provided by University of Utah Department of Atmospheric Sciences https://mesowest.utah.edu/

3. Bicycle sensors from the City of Munich Opendata Portal: https://opendata.muenchen.de/dataset/raddauerzaehlstellen-muenchen/resource/211e882d-fadd-468a-bf8a-0014ae65a393?view_id=11a47d6c-0bc1-4bfa-93ea-126089b59c3d

4. OpenStreetMap https://www.openstreetmap.org/ must also be referenced because the matrices where calculated using OpenStreetMap.

5. If you use any of the Maps you must reference both OpenStreetMap and Mapbox https://www.mapbox.com/.

6. For Shenzhen cite: https://github.com/lehaifeng/T-GCN

### Traffic Prediction Models

This document outlines the various models developed for traffic prediction. Each model is specifically designed to address unique aspects of traffic data analysis and prediction.

#### ARIMA_NN
- **Path**: `models.ARIMA_NN.ARIMA_NN`
- **Description**: Combines the ARIMA model with neural networks to capture both linear and non-linear patterns in traffic data. It is designed for univariate time series forecasting.
- **Parameters**:
  - `hidden_channels`: 1
  - `p`: 5
  - `d`: 1
  - `q`: 0

#### GCN_GRU
- **Path**: `models.GCN_GRU.GCN_GRU`
- **Description**: Leverages Graph Convolutional Networks (GCN) and Gated Recurrent Unit (GRU) for spatial-temporal traffic forecasting. It captures the spatial dependencies through GCN and temporal dynamics through GRU layers.
- **Parameters**:
  - `in_channels`: None
  - `hidden_channels`: 32
  - `num_gcn_layers`: 16
  - `num_rnn_layers`: 3
  - `dropout`: 0

#### GCN_GRU_BI
- **Path**: `models.GCN_GRU_BI.GCN_GRU_BI`
- **Description**: Extends the GCN_GRU model by introducing bidirectional GRU layers, enhancing its ability to understand complex temporal relationships in traffic data.
- **Parameters**:
  - Same as GCN_GRU

#### GCN_GRU_BI_Attention
- **Path**: `models.GCN_GRU_BI_Attention.GCN_GRU_BI_Attention`
- **Description**: Incorporates attention mechanisms into the GCN_GRU_BI model, allowing it to focus on critical temporal intervals for improved prediction accuracy.
- **Parameters**:
  - Same as GCN_GRU_BI

#### GCN_GRU_BI_Multi_Attention
- **Path**: `models.GCN_GRU_BI_Multi_Attention.GCN_GRU_BI_Multi_Attention`
- **Description**: Builds upon the GCN_GRU_BI_Attention model by adding multiple attention layers, further refining the model's focus on significant temporal features.
- **Parameters**:
  - Same as GCN_GRU_BI_Attention

#### GCN_GRU_TeacherForcing
- **Path**: `models.GCN_GRU_TeacherForcing.GCN_GRU_TeacherForcing`
- **Description**: Adopts teacher forcing in training the GCN_GRU model, leading to faster convergence and improved model performance by using the true past output as input.
- **Parameters**:
  - Same as GCN_GRU

#### GCN_LSTM
- **Path**: `models.GCN_LSTM.GCN_LSTM`
- **Description**: Integrates GCN with Long Short-Term Memory (LSTM) networks, aiming to exploit both spatial dependencies and long-term temporal patterns in traffic data.
- **Parameters**:
  - Same as GCN_GRU

#### GCN_LSTM_BI
- **Path**: `models.GCN_LSTM_BI.GCN_LSTM_BI`
- **Description**: Enhances the GCN_LSTM model by incorporating bidirectional LSTM layers, offering a more comprehensive analysis of temporal sequences.
- **Parameters**:
  - Same as GCN_LSTM

#### GCN_LSTM_BI_Attention
- **Path**: `models.GCN_LSTM_BI_Attention.GCN_LSTM_BI_Attention`
- **Description**: Adds an attention mechanism to the bidirectional GCN_LSTM model, improving its ability to prioritize important temporal segments for prediction.
- **Parameters**:
  - Same as GCN_LSTM_BI

#### GCN_LSTM_BI_Multi_Attention
- **Path**: `models.GCN_LSTM_BI_Multi_Attention.GCN_LSTM_BI_Multi_Attention`
- **Description**: Further extends the GCN_LSTM_BI_Attention model by utilizing multiple attention layers, enhancing the model's predictive accuracy by focusing on multiple relevant time periods simultaneously.
- **Parameters**:
  - Same as GCN_LSTM_BI_Attention

#### GCN_LSTM_BI_Multi_Attention_Weather
- **Path**: `models.GCN_LSTM_BI_Multi_Attention_Weather.GCN_LSTM_BI_Multi_Attention_Weather`
- **Description**: Incorporates weather data into the GCN_LSTM_BI_Multi_Attention model, acknowledging the impact of weather conditions on traffic patterns to improve forecasting accuracy.
- **Parameters**:
  - `in_channels`: None
  - `hidden_channels`: 64
  - `num_gcn_layers`: 64
  - `


### Sensor Predictions

<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/metr-la/sensors/sensor_716328/sensor_716328_predictions.jpg?raw=true" width="46%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/sensors/sensor_400073/sensor_400073_predictions.jpg?raw=true" width="46%" align="left">

<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

### Graph CFD Error Distribution

<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/acc_distribution.jpg?raw=true" width="17%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/r%C2%B2_distribution.jpg?raw=true" width="17%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/var_distribution.jpg?raw=true" width="17%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/mae_distribution.jpg?raw=true" width="17%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/rmse_distribution.jpg?raw=true" width="17%" align="left">
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/acc_heatmap.jpg?raw=true" width="17%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/r%C2%B2_heatmap.jpg?raw=true" width="17%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/var_heatmap.jpg?raw=true" width="17%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/mae_heatmap.jpg?raw=true" width="17%" align="left">
<img src="https://github.com/ThomasAFink/ST-GCN/blob/main/output/pems-bay/stats/rmse_heatmap.jpg?raw=true" width="17%" align="left">



