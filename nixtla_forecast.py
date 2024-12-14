import pandas as pd
from nixtla import NixtlaClient
from sklearn.metrics import mean_squared_error
from math import sqrt

# Carica il dataset
df = pd.read_csv('Renamed_y_Consumer_power.csv')

# Calcola l'indice per il 10% finale
split_point = int(len(df) * 0.9)

# Dividi il dataset in training e test
train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

# Inizializza il client di Nixtla con la tua chiave API
nixtla_client = NixtlaClient(api_key='nixak-gRhsLvQ95v1hkp6VYoTg4ZCa7ilMJjtyUDPnzL65pO8lOciYFCx1xz1zQmwL1fKZ5VTkiQSvuxm8XhfO')

# Previsioni one step ahead
predictions = []
for i in range(len(test_df)):
    current_train_df = pd.concat([train_df, test_df.iloc[:i]])
    forecast_df = nixtla_client.forecast(df=current_train_df, h=1, time_col='ds', target_col='y')
    predictions.append(forecast_df.iloc[0]['yhat'])
    train_df = current_train_df  # Ora il training set include il punto attuale del test set

# Calcola l'RMSE tra le previsioni e i valori reali
rmse = sqrt(mean_squared_error(test_df['y'], predictions))

# Mostra l'RMSE
print(f'RMSE: {rmse}')
