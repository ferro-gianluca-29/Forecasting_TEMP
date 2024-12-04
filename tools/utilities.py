import numpy as np
import matplotlib.pyplot as plt
import json
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import pickle
import numpy as np
import pandas as pd
import sys
from skforecast.utils import save_forecaster


def conditional_print(verbose, *args, **kwargs):
    """
    Prints provided arguments if the verbose flag is set to True.

    :param verbose: Boolean, controlling whether to print.
    :param args: Arguments to be printed.
    :param kwargs: Keyword arguments to be printed.
    """
    if verbose:
        print(*args, **kwargs)


def save_buffer(folder_path, df, target_column, size = 20, file_name = 'buffer.json'):
    """
    Saves a buffer of the latest data points to a JSON file.

    :param folder_path: Directory path where the file will be saved.
    :param df: DataFrame from which data will be extracted.
    :param target_column: Column whose data is to be saved.
    :param size: Number of rows to save from the end of the DataFrame.
    :param file_name: Name of the file to save the data in.
    """
    # Select the last rows with the specified column 
    buffer_df = df.iloc[-size:][[target_column]]  # Modified to keep as DataFrame
    
    # Convert the index timestamp column to string format if needed
    buffer_df.index = buffer_df.index.astype(str)

    # Serialize the dataframe to a JSON string
    try:
        buffer_json = buffer_df.to_json(orient='records')
        
        # Write the JSON string to the specified file
        with open(f"{folder_path}/{file_name}", 'w') as file:
            file.write(buffer_json)
        
        print(f"Data successfully saved to file {file_name}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

def load_trained_model(model_type, folder_name):
    """
    Loads a trained model and its configuration from the selected directory.

    :param model_type: Type of the model to load ('ARIMA', 'SARIMAX', etc.).
    :param folder_name: Directory from which the model and its details will be loaded.
    :return: A tuple containing the loaded model and its order (if applicable).
    """
    model = None
    best_order = None

    try:
        if model_type in ['ARIMA', 'SARIMAX', 'SARIMA']:
            with open(f"{folder_name}/model.pkl", "rb") as file:
                model = pickle.load(file)

            with open(f"{folder_name}/model_details_{model_type}.txt", "r") as file:
                for line in file:
                    if "Best Order" in line:
                        best_order_values = line.split(":")[1].strip().strip("()").split(", ")
                        best_order = tuple(map(int, best_order_values)) if best_order_values != [''] else None

    except FileNotFoundError:
        print(f"The folder {folder_name} does not contain a trained model.")
    except Exception as error:
        print(f"Error during model loading: {error}")

    return model, best_order

