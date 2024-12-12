import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier
from tools.time_series_analysis import  ljung_box_test



# pmdarima
import pmdarima
from pmdarima import ARIMA
from pmdarima import auto_arima

# skforecast
import skforecast
from skforecast.datasets import fetch_dataset
from skforecast.plot import set_dark_theme
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_sarimax
from skforecast.model_selection import grid_search_sarimax


from tqdm import tqdm
import pickle
from Predictors.Predictor import Predictor



class SARIMA_Predictor(Predictor):
    """
    A class used to predict time series data using Seasonal ARIMA (SARIMA) models.
    """

    def __init__(self, run_mode, target_column=None, period = 24,
                 verbose=False):
        """
        Constructs all the necessary attributes for the SARIMA_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param period: Seasonal period of the SARIMA model
        :param verbose: If True, prints detailed outputs during the execution of methods
        """

        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.period = period
        self.SARIMA_order = []
        self.model = None
        

    def train_model(self):
        """
        Trains a SARIMAX model using the training dataset and exogenous variables, if specified.

        :return: A tuple containing the trained model, validation metrics, and the index of the last training/validation timestep
        """
        try:    

            # CREATE SARIMA MODEL 

            d = 0
            D = 0

            # Selection of the model with best AIC score
            model = auto_arima(
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
            seasonal_order = model.seasonal_order

            # Create Fourier terms for weekly seasonality
            def create_fourier_terms(t, period, num_terms):
                terms = []
                for i in range(1, num_terms + 1):
                    terms.append(np.sin(2 * np.pi * i * t / period))
                    terms.append(np.cos(2 * np.pi * i * t / period))
                return np.column_stack(terms)

            num_fourier_terms = 4
            seasonality = 24  # Daily seasonality
            target_train = self.train[self.target_column]
            fourier_terms = create_fourier_terms(target_train, seasonality, num_fourier_terms)

            """model = auto_arima(target_train, 
                               #exogenous=fourier_terms[:n], 
                               seasonal=True, suppress_warnings=True)"""


            period = self.period  
            target_train = self.train[self.target_column]

            # Select directly the order (Comment if using the AIC search)
            """order = (2,1,1)
            seasonal_order = (2,0,1, 24)"""
            
            best_order = (order, seasonal_order)
            print(f"Best order found: {best_order}")
            

            self.SARIMA_order = best_order
            print("\nTraining the SARIMAX model...")

            forecaster = ForecasterSarimax( regressor=Sarimax(order=(2,1,1), seasonal_order=(2,0,1, 96)) )

            forecaster.fit(y=target_train)    

            residuals = forecaster.regressor.sarimax_res.resid    

            
               
            valid_metrics = None
            
            last_index = self.train.index[-1]
            # Running the LJUNG-BOX test for residual correlation
            #residuals = model.resid()
            #ljung_box_test(residuals)
            print("Model successfully trained.")

            return forecaster, valid_metrics, last_index
        
        except Exception as e:
                print(f"An error occurred during the model training: {e}")
                return None
        

    def test_model(self, forecaster, last_index, forecast_type, output_len, ol_refit = False, period = 24): 
        """
        Tests a SARIMAX model by performing one-step or multi-step ahead predictions, optionally using exogenous variables or applying refitting.

        :param model: The SARIMAX model to be tested
        :param last_index: Index of the last training/validation timestep
        :param forecast_type: Type of forecasting ('ol-one' for open-loop one-step ahead, 'cl-multi' for closed-loop multi-step)
        :param ol_refit: Boolean indicating whether to refit the model after each forecast
        :param period: The period for Fourier terms if set_fourier is true
        :return: A pandas Series of the predictions
        """
        try:    
            print("\nTesting SARIMA model...\n")
            
            self.forecast_type = forecast_type
            test = self.test
            self.steps_ahead = self.test.shape[0]
            full_data = pd.concat([self.train, self.test])
            

            if self.forecast_type == 'ol-one':
                steps = 1
            elif self.forecast_type == 'ol-multi':
                steps = output_len

            predictions = []
                           
            cv = TimeSeriesFold(
                        steps              = steps,
                        initial_train_size = len(self.train),
                        refit              = False,
                        fixed_train_size   = True,
                )
                
            _, predictions = backtesting_sarimax(
                        forecaster            = forecaster,
                        y                     = full_data[self.target_column],
                        cv                    = cv,
                        metric                = 'mean_absolute_error',
                        n_jobs                = "auto",
                        suppress_warnings_fit = True,
                        verbose               = True,
                        show_progress         = True
                     )
            
            predictions.rename(columns={'pred': self.target_column}, inplace=True)
            print("Model testing successful.")
            return predictions
                
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None 
        

    def plot_predictions(self, predictions):
        """
        Plots the SARIMA model predictions against the test data.

        :param predictions: The predictions made by the SARIMA model
        """
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='ARIMA')
        plt.title(f'SARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


    def save_model(self, path):
        # Save model
        #save_forecaster(self.model, f"{path}/SARIMA.joblib", verbose=False)
        # Save training info
        with open(f"{path}/model_details_SARIMA.txt", "w") as file:
            file.write(f"Training Info:\n")
            file.write(f"Best Order: {self.SARIMA_order}\n")
            file.write(f"End Index: {len(self.train)}\n")
            file.write(f"Target_column: {self.target_column}\n")
    
    def save_metrics(self, path, metrics):
        file_mode = "a" if os.path.exists(f"{path}/model_details_SARIMA.txt") else "w"
        # Save test info
        with open(f"{path}/model_details_SARIMA.txt", file_mode) as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n") 
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")

    
