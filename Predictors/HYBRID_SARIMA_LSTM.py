import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier
from tools.time_series_analysis import  ljung_box_test


from skforecast.utils import save_forecaster



import pmdarima
from pmdarima import auto_arima

from keras.layers import Dense,Flatten,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dropout, Dense, Reshape

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import pickle
import datetime
import sys

from Predictors.Predictor import Predictor

class Hybrid_Predictor(Predictor):
    """
    Class used to predict time series data using a hybrid approach of Seasonal ARIMA (SARIMA) and LSTM Network.

    NOTE: Since the Scaler is embedded in the Skforecast ForecasterRNN object, the --scalinig argument should NOT be given 
          to the command line parser when launching the code


    :param run_mode: The mode in which the predictor runs (e.g., 'train', 'test').
    :param input_len: The number of input time steps for the LSTM model.
    :param output_len: The number of output time steps to predict.
    :param target_column: The target column of the DataFrame to predict.
    :param period: Seasonal period of the SARIMA model, used for seasonal differencing.
    :param verbose: If True, prints detailed outputs during the execution of methods.
    :param forecast_type: The type of forecasting approach ('ol-one' for one-step ahead using online updates).
    """

    def __init__(self, run_mode, input_len, output_len, target_column=None, period = 24,
                 verbose=False, forecast_type='ol-one'):
        """
        Constructs all the necessary attributes for the SARIMA_Predictor object.
        """

        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.forecast_type = forecast_type
        self.period = period
        self.input_len = input_len
        self.output_len = output_len
        self.SARIMA_order = []
        self.lstm_model = None
        self.sarima_model = None
        

    def train_model(self, input_len, output_len):
        """
        Trains the SARIMA and LSTM model using the provided training dataset.

        This method fits a SARIMA model to the target time series and uses its residuals as input to train an LSTM model to capture nonlinear patterns and refine the predictions.

        :param input_len: The number of past observations the LSTM should consider for its input.
        :param output_len: The number of future time steps the LSTM is expected to predict.
        :return: A tuple containing the trained SARIMA model, the prediction DataFrame, and a fitted MinMaxScaler for data normalization.
        """
        try:    


            # CREATE SARIMA MODEL 

            d = 0
            D = 0

            # Selection of the model with best AIC score
            """model = auto_arima(
                        y=self.train[self.target_column],
                        start_p=0,
                        start_q=0,
                        max_p=4,
                        max_q=4,
                        seasonal=True,
                        m = self.period,
                        test='adf',
                        d=None,  # Let auto_arima determine the optimal 'd'
                        D=None,
                        trace=True,
                        error_action='warn',  # Show warnings for troubleshooting
                        suppress_warnings=False,
                        stepwise=True
                        )
            
            order = model.order
            seasonal_order = model.seasonal_order"""

            period = self.period  
            target_train = self.train[self.target_column]

            # Select directly the order (Comment if using the AIC search)
            order = (2,1,1)
            seasonal_order = (2,0,1, 24)
            
            best_order = (order, seasonal_order)
            print(f"Best order found: {best_order}")
            

            self.SARIMA_order = best_order
            print("\nTraining the SARIMAX model...")

            sarima_model = Sarimax( order = order,
                                        seasonal_order=seasonal_order,
                                        #maxiter = 500
                                        )
            
            sarima_model.fit(y=target_train)  

            # Save sarima model for later use 
            self.sarima_model = sarima_model  

            sarima_residuals = pd.DataFrame(sarima_model.sarimax_res.resid, columns=[self.target_column])
   

            # CREATE LSTM MODEL WITH KERAS FUNCTIONS

            def build_model(input_len, output_len, units=128, dropout_rate=0.2, learning_rate=0.001):
                
                optimizer = Adam(learning_rate=learning_rate)
                loss = 'mean_squared_error'
                input_shape = (input_len, 1)  
                
                model = Sequential()
                model.add(LSTM(units, activation='tanh', return_sequences=False, input_shape=input_shape))
                model.add(Dropout(dropout_rate)) 
                model.add(Dense(output_len, activation='linear'))
                model.add(Reshape((output_len, 1)))  

                model.compile(optimizer=optimizer, loss=loss)
                return model

            model = build_model(self.input_len, self.output_len)

            lstm_model = ForecasterRnn(
                                regressor = model,
                                levels = self.target_column,
                                transformer_series = MinMaxScaler(),
                                lags = self.input_len,
                                fit_kwargs={
                                    "epochs": 1,  # Number of epochs to train the model.
                                    "batch_size": 32,  # Batch size to train the model.
                                           },
                                    )    
            

            lstm_model.fit(sarima_residuals[[self.target_column]])

            # Save lstm model for later use 
            self.lstm_model = lstm_model

            current_time = datetime.datetime.now().strftime("%H_%M_%S")

            # FORECASTING LOOP

            steps = output_len
            predictions = []

            sarima_test_residuals = pd.Series(index=self.test.index, dtype=float)

            sarima_residuals_series = sarima_residuals[self.target_column].iloc[-self.input_len:]

            if self.forecast_type == 'ol-one':

                # Prima fase: finché non abbiamo abbastanza residui di test
                for i in tqdm(range(self.input_len), desc="Forecasting: Using last training timesteps..."):

                    # Forecast with SARIMA for one step
                    sarima_pred = sarima_model.predict(steps=1)

                    sarima_test_residuals.iloc[i] = self.test[self.target_column].iloc[i] - sarima_pred.values[0]

                    last_window_df = pd.concat([
                    sarima_residuals_series.iloc[-(self.input_len - i):],
                    sarima_test_residuals.iloc[:i]
                            ], axis=0).to_frame(name=self.target_column)
                    
                    last_window_df.index = pd.to_datetime(last_window_df.index)


                    # Forecast residual with LSTM for one step
                    lstm_pred = lstm_model.predict(steps=1, last_window=last_window_df)

                    # Combine predictions
                    combined_pred = sarima_pred.iloc[0, 0] + lstm_pred.iloc[0, 0]

                    # Append combined prediction
                    predictions.append(combined_pred)

                    # Update history with actual value for next iteration
                    actual_value = self.test[self.target_column].iloc[i]
                    sarima_model.append([actual_value], refit=False)


                # Seconda fase: quando i residui di test sono sufficienti
                for i in tqdm(range(self.input_len, len(self.test)), desc="Forecasting: Using predicted residual components"):

                    # Forecast with SARIMA for one step
                    sarima_pred = sarima_model.predict(steps=1)

                    sarima_test_residuals.iloc[i] = self.test[self.target_column].iloc[i] - sarima_pred.values[0]

                    recent_residuals = sarima_test_residuals.iloc[i - self.input_len: i]

                    last_window_df = recent_residuals.to_frame(name=self.target_column)
                    
                    last_window_df.index = pd.to_datetime(last_window_df.index)

                    # Forecast residual with LSTM for one step
                    lstm_pred = lstm_model.predict(steps=1, last_window=last_window_df)

                    # Combine predictions
                    combined_pred = sarima_pred.iloc[0, 0] + lstm_pred.iloc[0, 0]

                    # Append combined prediction
                    predictions.append(combined_pred)

                    # Update history with actual value for next iteration
                    actual_value = self.test[self.target_column].iloc[i]
                    sarima_model.append([actual_value], refit=False)

            prediction_index = self.test.index
            predictions_df = pd.DataFrame({self.target_column: predictions}, index=prediction_index)


            # fit scaler on train data to later scale test and predictions
            scaler = MinMaxScaler()
            temp_train = self.train.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
            scaler.fit(temp_train[temp_train.columns[0:temp_train.columns.shape[0] - 1]])

            return sarima_model, lstm_model, predictions_df, scaler

        except Exception as e:
                print(f"An error occurred during the model training: {e}")
                return None
        

    def test_model(self, forecaster, last_index, forecast_type, output_len, ol_refit = False, period = 24): 
        """ METHOD NOT USED FOR HYBRID PREDICTOR"""
        try:    
            
            
            return 
                
        
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None 
        

    def plot_predictions(self, predictions, test):
        """
        Plots the hybrid model predictions against the actual test data.

        :param predictions: The DataFrame containing the predicted values from the hybrid model.
        :param test: The actual test DataFrame used for evaluation.

        """
        test = test[self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='Hybrid Model')
        plt.title(f'Hybrid Model prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        # Save Sarima model
        with open(f"{path}/sarima_model.pkl", "wb") as file:
                            pickle.dump(self.sarima_model, file)
        # Save LSTM model
        save_forecaster(self.lstm_model, f"{path}/lstm.joblib", verbose=False)

    def save_metrics(self, path, metrics):
        # Save test info
        with open(f"{path}/model_details_HYBRID.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n") 
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")


    
