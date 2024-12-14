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



from keras.layers import LSTM, Dropout, Dense, Reshape
from keras.models import Sequential
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tqdm import tqdm

from skforecast.deep_learning import ForecasterRnn
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries

import warnings
warnings.filterwarnings("ignore")
from Predictors.Predictor import Predictor


class LSTM_Predictor(Predictor):
    """
    A class used to predict time series data using Long Short-Term Memory (LSTM) networks.
    """

    def _init_(self, run_mode, target_column=None, 
                 verbose=False, input_len=None, output_len=None, seasonal_model=False, set_fourier=False):
        """
        Constructs all the necessary attributes for the LSTM_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        :param input_len: Number of past observations to consider for each input sequence
        :param output_len: Number of future observations to predict
        :param seasonal_model: Boolean, if true include seasonal adjustments like Fourier features
        :param set_fourier: Boolean, if true use Fourier transformation on the data
        """
        super()._init_(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.input_len = input_len  
        self.output_len = output_len
        self.validation = validation
        self.model = None
        self.epochs = 1
        self.batch_size = 32
        self.learning_rate = 0.0001

        self.optimization_epochs = 1
        

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
            
            # CREATE MODEL  

            def build_model(input_len, output_len, units=128, dropout_rate=0.1, learning_rate=0.001):
                
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

            model = build_model(self.input_len, self.output_len)

            model.summary()

            lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)

            forecaster = ForecasterRnn(
                                regressor = model,
                                levels = self.target_column,
                                transformer_series = None,
                                lags = self.input_len,
                                fit_kwargs={
                                    "epochs": self.epochs,  # Number of epochs to train the model.
                                    "batch_size": self.batch_size,  # Batch size to train the model.
                                    "callbacks": [
                                                    lr_scheduler,
                                                ]  },
                                          )    
            
            #forecaster.fit(self.train[[self.target_column]])  

            #save model as an attribute for later use from external methods
            self.model = forecaster

            # uncomment this line to view the training matrix and check if the forecaster works as expected
            #X_train, y_train, _ = forecaster.create_train_X_y(series=self.train[[self.target_column]])

            return forecaster
        
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
    def test_model(self,forecaster):
        try:

            cv = TimeSeriesFold(
                steps=self.output_len,
                initial_train_size=None,
                refit=False,
            )

            mse, predictions = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=pd.concat([forecaster.last_window_, self.valid[self.target_column]]),
                levels=forecaster.levels,
                cv=cv,
                metric="mean_squared_error",
                verbose=False, # Set to True for detailed information
            )
                        
            print(f"BACKTESTING RMSE: {np.sqrt(mse['mean_squared_error'])}")
            
            return predictions
        
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        

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
           

    def plot_predictions(self, predictions):
        """
        Plots the LSTM model predictions against the test data.

        :param predictions: The predictions made by the LSTM model
        """
        test = self.test[self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='LSTM')
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