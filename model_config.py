import torch

models_map = {
        'ARIMA_NN': ("models.ARIMA_NN.ARIMA_NN", {
            "hidden_channels": 1,        
            "p": 5,
            "d": 1,
            "q": 0
        }),
        'SVR_NN': ("models.SVR_NN.SVR_NN", {
            "hidden_channels": 1,            
            "kernel": "rbf",
            "degree": 3,
            "C": 1.0,
            "epsilon": 0.1
        }),
        'GCN_GRU': ("models.GCN_GRU.GCN_GRU", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,            
            "dropout": 0,
        }),
        'GCN_GRU_BI': ("models.GCN_GRU_BI.GCN_GRU_BI", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,            
            "dropout": 0,
        }),
        'GCN_GRU_TeacherForcing': ("models.GCN_GRU_TeacherForcing.GCN_GRU_TeacherForcing", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,           
            "dropout": 0,
        }),
        'GCN_GRU_BI_Attention': ("models.GCN_GRU_BI_Attention.GCN_GRU_BI_Attention", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,            
            "dropout": 0,
        }),
        'GCN_GRU_BI_Multi_Attention': ("models.GCN_GRU_BI_Multi_Attention.GCN_GRU_BI_Multi_Attention", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,         
            "dropout": 0,
        }),
        'GCN_LSTM': ("models.GCN_LSTM.GCN_LSTM", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,          
            "dropout": 0,
        }),
        'GCN_LSTM_Peepholes': ("models.GCN_LSTM_Peepholes.GCN_LSTM_Peepholes", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,            
            "dropout": 0,
        }),
        'GCN_LSTM_TeacherForcing': ("models.GCN_LSTM_TeacherForcing.GCN_LSTM_TeacherForcing", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,            
            "dropout": 0,
        }),
        'GCN_LSTM_BI': ("models.GCN_LSTM_BI.GCN_LSTM_BI", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,            
            "dropout": 0,
        }),
        'GCN_LSTM_BI_TeacherForcing': ("models.GCN_LSTM_BI_TeacherForcing.GCN_LSTM_BI_TeacherForcing", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,           
            "dropout": 0,
        }),
        'GCN_LSTM_BI_Attention': ("models.GCN_LSTM_BI_Attention.GCN_LSTM_BI_Attention", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,            
            "dropout": 0,
        }),
        'GCN_LSTM_BI_Multi_Attention': ("models.GCN_LSTM_BI_Multi_Attention.GCN_LSTM_BI_Multi_Attention", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_rnn_layers": 3,          
            "dropout": 0,
        }),
        'GCN_Transformer': ("models.GCN_Transformer.GCN_Transformer", {
            "in_channels": None,
            "hidden_channels": 32,
            "num_gcn_layers": 16,
            "num_transformer_layers": 3,           
            "dropout": 0,
        })
}

def init_model(model_type, train_data, num_predictions, dropout=0):
    model_path, default_params = models_map[model_type]
    model_module, model_name = model_path.rsplit('.', 1)
    model_class = getattr(__import__(model_module, fromlist=[model_name]), model_name)

    # Set in_channels to the number of input features
    default_params["in_channels"] = train_data.size(1)
    default_params["num_predictions"] = num_predictions


    # Merge default params from models_map with provided params, with the latter taking precedence
    params = {
        **default_params  # Overwrite values from models_map with provided values
    }

    # Print params for debugging purposes
    print(f"Parameters being used: {params}")

    model = model_class(**params)

    # Post-processing for specific models
    if model_type == 'ARIMA_NN':
        train_data = train_data.to(dtype=torch.float32)
        numpy_train_data = train_data.numpy()
        model.arima.fit(numpy_train_data)

    elif model_type == 'SVR':
        train_data = train_data.to(dtype=torch.float32)
        numpy_train_data = train_data.numpy()
        model.svr.fit(numpy_train_data)

    return model
