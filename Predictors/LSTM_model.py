from datetime import datetime
import pandas as pd
import numpy as np
import datetime as datetime
import sys
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  
from skforecast.utils import save_forecaster


from sklearn.metrics import mean_squared_error


from sklearn.metrics import mean_squared_error

import random
import seaborn as sns

from keras.layers import LSTM, Dropout, Dense, Reshape
from keras.models import Sequential
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model


from tqdm import tqdm

from skforecast.deep_learning import ForecasterRnn
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries
import copy


import warnings
warnings.filterwarnings("ignore")
from Predictors.Predictor import Predictor


class LSTM_Predictor(Predictor):
    """
    A class used to predict time series data using Long Short-Term Memory (LSTM) networks.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False, input_len=None, output_len=None, validation = False):
        """
        Constructs all the necessary attributes for the LSTM_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        :param input_len: Number of past observations to consider for each input sequence
        :param output_len: Number of future observations to predict
        """
        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.input_len = input_len
        self.output_len = output_len
        self.validation = validation
        self.model = None
        self.epochs = 20
        self.batch_size = 1000
        self.learning_rate = 0.001
        

    def train_model(self):
        """
        Trains an LSTM model using the training and validation datasets.

        :param X_train: Input data for training
        :param y_train: Target variable for training
        :param X_valid: Input data for validation
        :param y_valid: Target variable for validation
        :return: A tuple containing the trained LSTM model and validation metrics
        """
        try:

            index_split = int(len(self.train) * 0.8)

            self.valid = self.train[index_split:]

            self.train = self.train[:index_split]

            for df in (self.train, self.valid, self.test):
                # Existing time features
                #df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
                #df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
                #df['week_of_year_sin'] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
                #df['week_of_year_cos'] = np.cos(2 * np.pi * df.index.isocalendar().week / 52)
                df['week_day_sin'] = np.sin(2 * np.pi * df.index.weekday / 7)
                df['week_day_cos'] = np.cos(2 * np.pi * df.index.weekday / 7)
                df['hour_day_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
                df['hour_day_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

                # Aggiunta delle caratteristiche per il giorno del mese
                df['day_sin'] = np.sin(2 * np.pi * df.index.day / df.index.days_in_month)
                df['day_cos'] = np.cos(2 * np.pi * df.index.day / df.index.days_in_month)

                # Rolling means
                #df['roll_mean_1_day'] = df[self.target_column].rolling(window=self.period, min_periods=1).mean()
                #df['roll_mean_7_day'] = df[self.target_column].rolling(window=self.period*7, min_periods=1).mean()

            # Aggiornamento dell'elenco delle caratteristiche esogene
            features = [
                self.target_column,
                #'month_sin', 
                #'month_cos',
                #'week_of_year_sin',
                #'week_of_year_cos',
                'week_day_sin',
                'week_day_cos',
                'hour_day_sin',
                'hour_day_cos',
                'day_sin',  # Aggiunta del seno del giorno
                'day_cos',  # Aggiunta del coseno del giorno
                #'roll_mean_1_day',
                #'roll_mean_7_day',
            
            ]

            self.features = features

            num_features = len(features)
            
            # CREATE MODEL  

            def build_model(input_len, output_len, units=128, dropout_rate=0.2, learning_rate=0.001):
                
                optimizer = Adam(learning_rate=learning_rate)
                loss = 'mean_squared_error'
                input_shape = (input_len, num_features)  
                
                model = Sequential()
                model.add(LSTM(units, activation='tanh', return_sequences=False, input_shape=input_shape))
                model.add(Dropout(dropout_rate))
                model.add(Dense(output_len, activation='linear'))
                model.add(Reshape((output_len, 1)))  

                model.compile(optimizer=optimizer, loss=loss, 
                              metrics=[RootMeanSquaredError()])
                return model

            model = build_model(self.input_len, self.output_len, learning_rate = self.learning_rate)

            model.summary()

            callbacks = [
                        EarlyStopping(monitor='val_loss', patience=80, verbose=1, mode='min', restore_best_weights=True),
                    ]

            X_train, y_train = self.data_windowing(self.train[features])
            X_valid, y_valid = self.data_windowing(self.valid[features])
            
            
            if self.validation:
                # Select validation method by uncommenting the desired line
                validated_model_weights = self.repeated_holdout_validation(model, nreps=10)
                #validated_model_weights = self.prequential_validation(model)

                # Train the model based on validation parameters

                model = build_model(self.input_len, self.output_len)
                model.set_weights(validated_model_weights)
                history = model.fit(X_train, y_train, 
                               epochs=self.epochs,  
                               batch_size=self.batch_size)
                
                

            else:
                history = model.fit(
                                    X_train, y_train,
                                    epochs=self.epochs,  # Numero di epoche per il training
                                    batch_size=self.batch_size,  # Dimensione del batch, può essere regolata in base alle necessità
                                    validation_data=(X_valid, y_valid),
                                    callbacks=callbacks,  # Inclusione delle callback per il monitoraggio e il salvataggio
                                    verbose=1  # Per visualizzare il progresso durante il training
                                )
                
                # Estrazione dei valori di loss di training e validazione dall'oggetto history
                train_loss = history.history['loss']
                val_loss = history.history['val_loss']
                epochs = range(1, len(train_loss) + 1)

                # Creazione del grafico
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
                plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)

                plt.show()

            #save model as an attribute for later use from external methods
            self.model = model

            return model
        
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
    def test_model(self,model):
        try:
            X_test, y_test = self.data_windowing(self.test[self.features])
            predictions = model.predict(X_test)
            predictions = predictions.flatten()
            
            return predictions, y_test
        
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None

    def data_windowing(self, df):
        stride = 1 if self.output_len == 1 else self.output_len
        X, y = [], []
        indices = []

        if len(df) < self.input_len + self.output_len:
            print("Data is too short for creating windows")
            return None
        else:
            for i in range(0, len(df) - self.input_len - self.output_len + 1, stride):
                X.append(df.iloc[i:i + self.input_len].values)
                y.append(df[self.target_column].iloc[i + self.input_len:i + self.input_len + self.output_len].values)
                indices.append(i)

        # Conversione in array
        X, y = np.array(X), np.array(y)

        # Reshape di X per includere tutte le feature
        X = np.reshape(X, (X.shape[0], self.input_len, -1))  # -1 qui farà in modo che numpy calcoli automaticamente il numero corretto di features

        return X, y
        
    def repeated_holdout_validation(self, base_model, nreps=10):
      
        train_size = int(0.6 * len(self.train))
        val_size = int(0.1 * len(self.train))

        target_train = self.train[[self.target_column]]

        weights_history = []
        
        rmse_values = []

        for i in range(nreps):
            # Create a new instance of the forecaster 
            model = clone_model(base_model)
            # Control if there's enough space for training and validation windows
            t_min = train_size
            t_max = len(self.train) - val_size
            if t_min >= t_max:
                raise ValueError("Not enough data to allocate both training and validation windows.")

            # Choose a random t so that it is possible to have both training and validation windows
            t = random.randint(t_min, t_max)
        
            # Create training subset
            train_subset = target_train.iloc[t-train_size:t]
            # Create validation subset
            val_subset = target_train.iloc[t:t+val_size]

            X_train, y_train = self.data_windowing(train_subset)
            X_val, y_val = self.data_windowing(val_subset)

            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error', 
                              metrics=[RootMeanSquaredError()])
            
            # Train the model on training subset
            history = model.fit(X_train, y_train, 
                               epochs=self.epochs,  
                               batch_size=self.batch_size) 
            
            # Test on the validation subset
            predictions = model.predict(X_val)
            # Compute RMSE
            mse = mean_squared_error(y_val.flatten(), predictions.flatten())
            rmse = np.sqrt(mse)

            print(f"RMSE: {rmse}")
            rmse_values.append(rmse)

            # Collect weights after training
            session_weights = [layer.get_weights() for layer in model.layers]
            weights_history.append(session_weights)

        # Compute average and variance values of weights per layer
        average_weights = []
        variance_weights = []
        for layer_index in range(len(weights_history[0])):  # Iterate on each layer
            # Extract layer weights from each training session
            layer_weights = [session[layer_index] for session in weights_history]
            # Compute the average
            mean_weights = [np.mean([weights[j] for weights in layer_weights], axis=0) for j in range(len(layer_weights[0]))]
            average_weights.append(mean_weights)
            # Compute the variance
            var_weights = [np.var([weights[j] for weights in layer_weights], axis=0) for j in range(len(layer_weights[0]))]
            variance_weights.append(var_weights)

        mean_rmse = np.mean(rmse_values)
        print(f"mean validation rmse: {mean_rmse}")

        validated_model = clone_model(base_model)

        # Set the average weights on the model
        for layer_index, layer in enumerate(validated_model.layers):
            layer.set_weights(average_weights[layer_index])

        validated_model_weights = validated_model.get_weights()

        return validated_model_weights
    

    def prequential_validation(self, model, fold_dim=0.1):
        
        target_train = self.train[[self.target_column]]
        data_length = len(target_train)
        fold_size = int(data_length * fold_dim)
        
        # Calcola il numero di fold basato sulla dimensione specificata
        num_folds = max(1, data_length // fold_size)
        print(f"number of folds: {num_folds}")
        
        # Crea l'array dei fold
        folds = np.repeat(np.arange(1, num_folds + 1), fold_size)
        # Gestisce il caso in cui i fold non coprano esattamente tutto il dataset
        additional_elements = data_length % fold_size
        if additional_elements > 0:
            folds = np.append(folds, np.full(additional_elements, num_folds))
        
        rmse_values = []

        # L'ultimo fold non verrà usato come training set
        for fold_index in range(num_folds - 1):
            train_mask = folds <= fold_index + 1
            valid_mask = folds == fold_index + 2
            train_data = target_train[train_mask]
            valid_data = target_train[valid_mask]

            X_train, y_train = self.data_windowing(train_data)
            X_valid, y_valid = self.data_windowing(valid_data)

            history = model.fit(X_train, y_train, 
                               epochs=self.epochs,  
                               batch_size=self.batch_size)

            predictions = model.predict(X_valid)
            mse = mean_squared_error(y_valid.flatten(), predictions.flatten())
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)

        # Plot the RMSE values over iterations
        plt.figure(figsize=(10, 5))
        plt.plot(rmse_values, marker='o', linestyle='-')
        plt.title('Andamento RMSE attraverso le iterazioni di validazione incrociata')
        plt.xlabel('Numero di Iterazione')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.show()

        model_weights = model.get_weights()

        return model_weights
    

    def unscale_data(self, predictions, y_test, folder_path):
        
        """
        Unscales the predictions and test data using the scaler saved during model training.

        :param predictions: The scaled predictions that need to be unscaled
        :param y_test: The scaled test data that needs to be unscaled
        :param folder_path: Path to the folder containing the scaler object
        """
        # Load scaler for unscaling data
        with open(f"{folder_path}/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        
        # Unscale predictions
        predictions = predictions.to_numpy().reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions) 
        predictions = predictions.flatten() 
        # Unscale test data
        y_test = pd.DataFrame(y_test)
        y_test = scaler.inverse_transform(y_test)
        y_test = pd.Series(y_test.flatten())

        return predictions, y_test                                
           

    def plot_predictions(self, predictions, y_test):
        """
        Plots the LSTM model predictions against the test data.

        :param predictions: The predictions made by the LSTM model
        """
        test = self.test[self.target_column]
        plt.plot(range(len(y_test)), y_test.flatten(), 'b-', label='Test Set')
        plt.plot(range(len(y_test)), predictions, 'k--', label='LSTM')
        plt.title(f'LSTM prediction for Malaysia hourly load dataset', fontsize=18)
        plt.xlabel('Time series index', fontsize=14)
        plt.ylabel('Scaled value', fontsize=14)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        save_forecaster(self.model, f"{path}/LSTM.joblib", verbose=False)

    def save_metrics(self, path, metrics):
        # Save test info
        with open(f"{path}/model_details_LSTM.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n") 
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")



