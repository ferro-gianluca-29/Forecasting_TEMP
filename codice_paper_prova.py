import os
import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from collections import namedtuple

from statistics import NormalDist

import pmdarima as pm

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

from datetime import datetime

# Глобальная переменная для хранения настроек фигур
FigureSettings = namedtuple('FigureSettings', ['figWidth', 'figHeight', 'dpi', 'labelXSize', 'labelYSize',
                                               'tickMajorLabelSize', 'tickMinorLabelSize', 'tickXLabelRotation',
                                               'markerSize', 'legendFontSize', 'titleFontSize', 'boxTextSize'])
fset = FigureSettings
fset.figWidth = 10
fset.figHeight = 6
fset.dpi = 300
fset.labelXSize = 14
fset.labelYSize = 14
fset.tickMajorLabelSize = 12
fset.tickMinorLabelSize = 10
fset.tickXLabelRotation = 30
fset.markerSize = 5
fset.legendFontSize = 14
fset.titleFontSize = 14
fset.boxTextSize = 14

# Путь к корневой папке проекта
dir_name = os.path.dirname(__file__)

# Горизонт прогноза в часах
forecast_horizon = 0

# Датафреймы для обучающей и тестовой выборки
df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_date = pd.DataFrame()

# Датафреймы для методов LSTM и XGBoost

X_train = pd.DataFrame()
y_train = pd.DataFrame()
X_test = pd.DataFrame()
y_test = pd.DataFrame()

look_back = 0
internal_units = 0
batch_size = 0
epochs = 0

scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

plot_show = False

df_statistic = pd.DataFrame({'Method': pd.Series(dtype='str'),
                             'R2': pd.Series(dtype='str'),
                             'MSE': pd.Series(dtype='str'),
                             'RMSE': pd.Series(dtype='str'),
                             'MAE': pd.Series(dtype='str'),
                             'MAPE': pd.Series(dtype='str'),
                             'T.Time': pd.Series(dtype='str'),
                             'P.Time': pd.Series(dtype='str')})

df_predictions = pd.DataFrame(columns=['DT'])

# Предварительная обработка данных
def data_preprocessing(report_file_name: str, column_name: str, data_file_name: str,
                       start_date_train, end_date_train, start_date_test, end_date_test, add_hour):
    """
    Функция предварительной обработки данных
    :param report_file_name: string - Полный путь к входному файлу с данными
    :param column_name: string - Имя колонки, которая будет использована как источник данных
    :param data_file_name: string - Полный путь к выходному файлу с данными
    """
    start_time = time.time()

    # Чтение данных
    df = pd.read_csv(report_file_name, parse_dates=['DT'])

    # Формируем новый датасет только из двух столбцов - дата (DS) и значение (ColumnName)
    df_new = df[['DT', column_name]]
    df_new = df_new.set_index('DT')

    # Передискретизация / Resample
    # W - неделя
    # D - календарный день
    # H - час
    # T - минута
    # S - секунда

    df_resample = pd.DataFrame(data=df_new[column_name].resample('H').mean())

    # Фильтрация по временному кадру
    if add_hour:
        df_resample = df_resample[(df_resample.index >= start_date_train) & (df_resample.index <= end_date_test)]
    else:
        df_resample = df_resample[(df_resample.index >= start_date_train) & (df_resample.index < end_date_test)]

    # Изменяем заголовок столбца
    df_resample.columns = ['Measurement']
    # Сохранить новый датафрейм в файл
    df_resample.to_csv(data_file_name)

    # Визуализация
    values = df_resample['Measurement']
    fig, ax = plt.subplots(figsize=(fset.figWidth, fset.figHeight))
    fig.set_dpi(fset.dpi)
    ax.plot(values)
    ax.axvline(x=start_date_test, color='g', alpha=0.5)
    ax.axvline(x=end_date_test, color='g', alpha=0.5)

    ax.axvspan(start_date_test,
               end_date_test, color='g', alpha=0.2)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

    avg_date_train = start_date_train + (end_date_train - start_date_train) / 2
    ax.text(avg_date_train, 8.5, "   Train Data   ",
            ha="center", va="center", size=fset.boxTextSize, bbox=bbox_props)
    avg_date_test = start_date_test + (end_date_test - start_date_test) / 2
    ax.text(avg_date_test, 8.5, "Test Data",
            ha="center", va="center", size=fset.boxTextSize, bbox=bbox_props)

    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.title('History Horizon', style='italic', fontsize=fset.labelXSize)
    plt.ylabel('Measurement', style='italic', fontsize=fset.labelYSize)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis='both', which='major', labelsize=fset.tickMajorLabelSize)
    ax.tick_params(axis='both', which='minor', labelsize=fset.tickMinorLabelSize)

    plt.setp(ax.get_xticklabels(), rotation=fset.tickXLabelRotation)
    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), 'dati', 'Fig1.png')), dpi=fset.dpi,
                bbox_inches='tight')

    plt.ylim(2, 9)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    if plot_show:
        plt.show()


def train_test_split(data_file_name: str, start_date_train, end_date_train, start_date_test, end_date_test, add_hour):

    global df_train
    global df_test
    global df_date
    global forecast_horizon
    global df_predictions

    df = pd.read_csv(data_file_name, parse_dates=['DT'])
    df.rename(columns={'DT': 'ds', 'Measurement': 'y'}, inplace=True)

    # Обучающая и тестовая выборки
    if add_hour:
        df_train = df[(df['ds'] >= start_date_train) & (df['ds'] < end_date_train)]
        df_test = df[(df['ds'] >= start_date_test) & (df['ds'] <= end_date_test)]
    else:
        df_train = df[(df['ds'] >= start_date_train) & (df['ds'] < end_date_train)]
        df_test = df[(df['ds'] >= start_date_test) & (df['ds'] < end_date_test)]

    data_range = pd.date_range(start_date_test, end_date_test, freq='H')

    if not add_hour:
        df_date = data_range.to_frame().iloc[:-1, :]
    else:
        df_date = data_range.to_frame()

    df_date.columns = ['ds']
    df_predictions['DT'] = df_test['ds']
    df_predictions['Y'] = df_test['y']
    df_predictions.reset_index(drop=True, inplace=True)

    # Расчет горизонта прогноза
    duration = end_date_test - start_date_test
    duration_seconds = duration.total_seconds()
    forecast_horizon = int(divmod(duration_seconds, 3600)[0])

    if add_hour:
        forecast_horizon = forecast_horizon + 1


def dataset_generation(data_file_name,
                       data_file_name_nn,
                       data_file_name_nn_scaler,
                       data_file_name_nn_scaler_train,
                       data_file_name_nn_scaler_test,
                       start_date_test):
    global X_train
    global y_train
    global X_test
    global y_test

    global look_back

    global scalerX
    global scalerY

    df = pd.read_csv(data_file_name, parse_dates=['DT'])
    df.rename(columns={'DT': 'Timestamp', 'Measurement': 'y'}, inplace=True)

    data = []
    for i in range(len(df) - look_back):
        data.append(
            {
                'y': df.iloc[i + look_back, 1],
                'X': df.iloc[i:(i + look_back), 1].values
            })

    df_ = pd.DataFrame(data)

    # Генерация имен колонок для нового датафрейма
    col_names = []
    for i in reversed(range(look_back)):
        col_names.append(f'L{i + 1}')

    y = pd.DataFrame(df_['y'].to_list(), columns=['y'])
    X = pd.DataFrame(df_['X'].to_list(), columns=col_names)

    # Датафрейм с временными метками
    T = df['Timestamp'][look_back:]
    T.reset_index(drop=True, inplace=True)

    dataset = pd.concat([T, pd.DataFrame(y, columns=['y']), pd.DataFrame(X, columns=col_names)], axis=1)
    dataset.to_csv(data_file_name_nn)

    # Масштабирование
    X = scalerX.fit_transform(X)
    y = scalerY.fit_transform(y)

    # Итоговый датасет
    dataset = pd.concat([T, pd.DataFrame(y, columns=['y']), pd.DataFrame(X, columns=col_names)], axis=1)
    dataset.to_csv(data_file_name_nn_scaler)

    # Разбивка на обучающую и тестовую выборки
    train = dataset[dataset['Timestamp'] < start_date_test]
    train.to_csv(data_file_name_nn_scaler_train)
    test = dataset[dataset['Timestamp'] >= start_date_test]
    test.to_csv(data_file_name_nn_scaler_test)
    # Сброс индекса для тестовой выборки
    test.reset_index(drop=True, inplace=True)

    # Разбивка данных
    X_train = train.iloc[:, 2:]
    y_train = train.iloc[:, 1:2]
    X_test = test.iloc[:, 2:]
    y_test = test.iloc[:, 1:2]


def get_confidence_intervals(data, confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((len(data) - 1) ** .5)
    lower = data - h*2
    upper = data + h*2
    return lower, upper


# Оценка производительности метода
def performance_evaluation(y_true: pd.DataFrame,
                           y_pred: pd.DataFrame,
                           method_name: str,
                           train_time: str,
                           pred_time: str):

    global df_statistic

    r2_value = r2(y_true, y_pred)
    mse_value = mse(y_true, y_pred)
    rmse_value = mse(y_true, y_pred, squared=False)
    mae_value = mae(y_true, y_pred)
    mape_value = mape(y_true, y_pred)

    print(f'\n=================================================================')
    print(f'Performance Evaluation for {method_name}')
    print(f'R2: {r2_value:.3f}')
    print(f'MSE: {mse_value:.3f}')
    print(f'RMSE: {rmse_value:.3f}')
    print(f'MAE: {mae_value:.3f}')
    print(f'MAPE: {mape_value:.3f}')

    #st = {'Method': method_name,
    #      'R2': f'{r2_value:.3f}',
    #      'MSE': f'{mse_value:.3f}',
    #      'RMSE': f'{rmse_value:.3f}',
    #      'MAE': f'{mae_value:.3f}',
    #      'MAPE': f'{mape_value:.3f}',
    #      'T.Time': train_time,
    #      'P.Time': pred_time}
    #df_statistic = df_statistic.append(st, ignore_index=True)

    item = [[method_name, f'{r2_value:.3f}', f'{mse_value:.3f}', f'{rmse_value:.3f}', f'{mae_value:.3f}',
             f'{mape_value:.3f}', train_time, pred_time]]
    cols = ['Method', 'R2', 'MSE', 'RMSE', 'MAE', 'MAPE', 'T.Time', 'P.Time']

    df = pd.DataFrame(item, columns=cols)
    df.reset_index(drop=True, inplace=True)
    df_statistic.reset_index(drop=True, inplace=True)

    df_statistic = pd.concat([df_statistic, df], axis=0, ignore_index=True)


def performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper, method_name, fig_file_name):
    global plot_show

    fig, ax = plt.subplots(figsize=(fset.figWidth, fset.figHeight))
    fig.set_dpi(fset.dpi)
    ax.plot(x_date, y_pred, '-')
    ax.plot(x_date, y_true, 'o', color='tab:brown', markersize=fset.markerSize)
    ax.fill_between(x_date, ypred_lower, ypred_upper, alpha=0.2)
    ax.legend(['Predicted Value', 'Actual Value'], fontsize=fset.legendFontSize, loc='lower right')

    plt.grid(color='0.75', linestyle='--', linewidth=0.5)
    plt.title(f'{method_name.upper()} Forecast Horizon', style='italic', fontsize=fset.titleFontSize)
    plt.ylabel('Measurement', style='italic', fontsize=fset.labelYSize)

    plt.ylim(2, 9)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis='both', which='major', labelsize=fset.tickMajorLabelSize)
    ax.tick_params(axis='both', which='minor', labelsize=fset.tickMinorLabelSize)
    plt.setp(ax.get_xticklabels(), rotation=fset.tickXLabelRotation)
    plt.savefig(fname=os.path.realpath(os.path.join(os.path.dirname(__file__), 'img', fig_file_name)),
                dpi=fset.dpi, bbox_inches='tight')

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    if plot_show:
        plt.show()



# Прогнозирование с помощью метода SARIMA
def sarima_forecasting():
    # Данные
    data = df_train['y']

    # Augmented Dickey-Fuller test
    adf = adfuller(data, autolag='AIC')
    print(f'Augmented Dickey-Fuller Test for SARIMA method\n')
    print("1. ADF : ", adf[0])
    print("2. P-Value : ", adf[1])
    print("3. Num Of Lags : ", adf[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", adf[3])
    print("5. Critical Values :")
    for key, val in adf[4].items():
        print("\t", key, ": ", val)

    start_time = time.time()
    # Подбор параметров
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html#pmdarima.arima.auto_arima
    print('\nAutoARIMA\n')
    m = pm.arima.auto_arima(data,
                            start_p=1,
                            start_q=1,
                            max_p=3,
                            max_q=3,
                            m=24,
                            start_P=0,
                            seasonal=True,
                            d=None,
                            D=1,
                            test='adf',
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

    train_time = f'{(time.time() - start_time): .3f}'

    print(m.summary())

    m = SARIMAX(data, order=(3, 0, 2), seasonal_order=(0, 1, 2, 24),
                enforce_stationarity=False, enforce_invertibility=False).fit()

    print(m.summary())

    start_time = time.time()
    forecast = m.get_forecast(steps=forecast_horizon, signal_only=True)
    pred_time = f'{(time.time() - start_time): .3f}'

    forecast_interval = forecast.conf_int()

    x_date = df_date['ds']
    y_true = df_test['y']
    y_pred = forecast.predicted_mean
    ypred_lower = forecast_interval.iloc[:, 0]
    ypred_upper = forecast_interval.iloc[:, 1]

    performance_evaluation(y_true, y_pred, 'SARIMA', train_time, pred_time)
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper, 'SARIMA', 'Fig3.png')

    df_ = pd.DataFrame(y_pred)
    df_.reset_index(drop=True, inplace=True)
    df_predictions['SARIMA'] = df_


# Прогнозирование с помощью Holt-Winters Exponential Smoothing
def holtwinters_forecasting():
    # Данные
    data = df_train['y']

    start_time = time.time()
    m = ExponentialSmoothing(data, seasonal='add', seasonal_periods=24).fit()
    train_time = f'{(time.time() - start_time): .3f}'

    start_time = time.time()
    forecast = m.forecast(steps=forecast_horizon)
    pred_time = f'{(time.time() - start_time): .3f}'

    x_date = df_date['ds']
    y_true = df_test['y']
    y_pred = forecast.values
    ypred_lower, ypred_upper = get_confidence_intervals(forecast.values)

    performance_evaluation(y_true, y_pred, 'Holt-Winters Exponential Smoothing', train_time, pred_time)
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper,
                              'Holt-Winters Exponential Smoothing', 'Fig4.png')

    df_predictions['HOLT WINTER'] = y_pred


# Прогнозирование с помощью ETS Model (модификация Holt-Winters Exponential Smoothing)
def etsmodel_forecasting():
    # Данные
    data = df_train['y']

    # Создаем модель
    start_time = time.time()
    m = ETSModel(endog=data, seasonal='add', seasonal_periods=24).fit()
    train_time = f'{(time.time() - start_time): .3f}'

    start_time = time.time()
    prediction = m.get_prediction(start=df_test.index[0], end=df_test.index[-1])
    pred_time = f'{(time.time() - start_time): .3f}'

    ci = prediction.pred_int(alpha=.05)  # confidence interval 0.95
    forecast = pd.concat([prediction.predicted_mean, ci], axis=1)
    forecast.columns = ['yhat', 'yhat_lower', 'yhat_upper']

    x_date = df_date['ds']
    y_true = df_test['y']
    y_pred = forecast['yhat']
    ypred_lower = forecast['yhat_lower']
    ypred_upper = forecast['yhat_upper']

    performance_evaluation(y_true, y_pred, 'ETS Model', train_time, pred_time)
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper,
                              'ETS Model', 'Fig5.png')

    df_ = pd.DataFrame(y_pred)
    df_.reset_index(drop=True, inplace=True)
    df_predictions['ETS MODEL'] = pd.DataFrame(df_['yhat'])


def lstm_forecasting(units, look_back, epochs, batch_size):

    global X_train
    global y_train
    global X_test
    global y_test

    start_time = time.time()
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    train_time = f'{(time.time() - start_time):.3f}'

    print('\nLSTM model summary')
    print(model.summary())

    start_time = time.time()
    yhat = model.predict(X_test, verbose=0)
    pred_time = f'{(time.time() - start_time):.3f}'

    yhat = pd.DataFrame(yhat)

    x_date = df_date['ds']
    y_true = scalerY.inverse_transform(y_test)
    y_pred = yhat.values
    y_pred = scalerY.inverse_transform(y_pred)

    ypred_lower, ypred_upper = get_confidence_intervals(pd.DataFrame(y_pred).values.flatten())

    performance_evaluation(y_true, y_pred, 'LSTM (real test data)', train_time, pred_time)
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper,
                              'Long short-term memory (real test data)', 'Fig6.png')

    df_predictions['LSTM RTD'] = y_pred

    # Synthetic test data generation
    start_time = time.time()
    x_train = X_train.iloc[-1:]
    x_train = x_train.reset_index(drop=True)
    y_pred = model.predict(x_train, verbose=0)

    col_names = []
    for i in reversed(range(look_back)):
        col_names.append(f'L{i + 1}')

    y_pred_new = []
    for i in range(len(y_test)):
        x_list = x_train.iloc[:, 1:].values.tolist()
        y_list = y_pred.tolist()
        x_train = pd.DataFrame(x_list[0] + y_list[0])
        x_train = x_train.transpose()
        x_train.columns = col_names
        y_pred = model.predict(x_train, verbose=0)
        y_pred_new.append(y_pred.tolist()[0])

    pred_time = f'{(time.time() - start_time):.3f}'

    x_date = df_date['ds']
    y_true = scalerY.inverse_transform(y_test)
    y_pred = y_pred_new
    y_pred = scalerY.inverse_transform(y_pred)

    ypred_lower, ypred_upper = get_confidence_intervals(pd.DataFrame(y_pred).values.flatten())

    performance_evaluation(y_true, y_pred, 'LSTM (synthetic test data)', train_time, pred_time)
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper,
                              'Long short-term memory (synthetic test data)', 'Fig7.png')

    df_predictions['LSTM STD'] = y_pred


def xgboost_forecasting(max_depth, learning_rate, n_estimators, gamma):
    global X_train
    global y_train
    global X_test
    global y_test
    global look_back

    print('\nXGBoost Model summary')

    start_time = time.time()
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         gamma=gamma)

    model.fit(X_train, y_train)
    train_time = f'{(time.time() - start_time):.3f}'

    print(model)

    start_time = time.time()
    yhat = model.predict(X_test)
    pred_time = f'{(time.time() - start_time):.3f}'

    yhat = pd.DataFrame(yhat)

    x_date = df_date['ds']
    y_true = scalerY.inverse_transform(y_test)
    y_pred = yhat.values
    y_pred = scalerY.inverse_transform(y_pred)

    ypred_lower, ypred_upper = get_confidence_intervals(pd.DataFrame(y_pred).values.flatten())

    performance_evaluation(y_true, y_pred, 'XGBoost (real test data)', train_time, pred_time)
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper,
                              'XGBoost (real test data)', 'Fig8.png')

    df_predictions['XGBoost RTD'] = y_pred

    # Synthetic test data generation
    start_time = time.time()
    x_train = X_train.iloc[-1:]
    x_train = x_train.reset_index(drop=True)
    y_pred = model.predict(x_train)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.values.tolist()

    col_names = []
    for i in reversed(range(look_back)):
        col_names.append(f'L{i + 1}')

    y_pred = model.predict(x_train)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.values

    y_pred_new = []
    for i in range(len(y_test)):
        x_list = x_train.iloc[:, 1:].values.tolist()
        y_list = y_pred.tolist()
        x_train = pd.DataFrame(x_list[0] + y_list[0])
        x_train = x_train.transpose()
        x_train.columns = col_names
        y_pred = model.predict(x_train)
        y_pred = pd.DataFrame(y_pred)
        y_pred = y_pred.values
        y_pred_new.append(y_pred.tolist()[0])

    pred_time = f'{(time.time() - start_time):.3f}'
    x_date = df_date['ds']
    y_true = scalerY.inverse_transform(y_test)
    y_pred = y_pred_new
    y_pred = scalerY.inverse_transform(y_pred)

    ypred_lower, ypred_upper = get_confidence_intervals(pd.DataFrame(y_pred).values.flatten())

    performance_evaluation(y_true, y_pred, 'XGBoost (synthetic test data)', train_time, pred_time)
    performance_visualisation(x_date, y_true, y_pred, ypred_lower, ypred_upper,
                              'XGBoost (synthetic test data)', 'Fig9.png')

    df_predictions['XGBoost STD'] = y_pred


if __name__ == '__main__':
    print(f'CALCULATION STARTED AT {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    start = time.time()
    plot_show = False

    # File names

    dir_name = '.'

    src_file_name = os.path.realpath(os.path.join(dir_name, 'dati', 'initialdataset.csv'))
    dat_file_name = os.path.realpath(os.path.join(dir_name, 'dati', 'data.csv'))
    dat_file_name_nn = os.path.realpath(os.path.join(dir_name, 'dati', 'data_nn.csv'))
    dat_file_name_nn_scaler = os.path.realpath(os.path.join(dir_name, 'dati', 'data_nn_scaler.csv'))
    dat_file_name_nn_scaler_train = os.path.realpath(os.path.join(dir_name, 'dati', 'data_nn_scaler_train.csv'))
    dat_file_name_nn_scaler_test = os.path.realpath(os.path.join(dir_name, 'dati', 'data_nn_scaler_test.csv'))
    rep_file_name = os.path.realpath(os.path.join(dir_name, 'dati', 'forecasting-report.txt'))
    prd_file_name = os.path.realpath(os.path.join(dir_name, 'dati', 'predictions.csv'))
    prm_file_name = os.path.realpath(os.path.join(dir_name, 'dati', 'parameters.txt'))

    # Data column name
    col_name = 'Influent'

    # Data Preprocessing
    start_date_train = dt.datetime(2022, 12, 12)
    end_date_train = dt.datetime(2022, 12, 29)
    start_date_test = dt.datetime(2022, 12, 29)
    end_date_test = dt.datetime(2022, 12, 31)
    add_hour = False
    

    data_preprocessing(src_file_name, col_name, dat_file_name,
                      start_date_train, end_date_train, start_date_test, end_date_test, add_hour)

    # Разбивка датасета на тестовую и обучающую выборки
    train_test_split(dat_file_name, start_date_train, end_date_train, start_date_test, end_date_test, add_hour)


    # SARIMA Forecasting
    sarima_forecasting()

    # Holt-Winters ES Forecasting
    holtwinters_forecasting()

    # ETS Model
    etsmodel_forecasting()

    # LSTM Dataset generation

    look_back = 48  # History interval
    internal_units = 32  # The number of neurons
    batch_size = 16  # Batch size
    epoch_count = 200  # Epochs count

    dataset_generation(dat_file_name,
                       dat_file_name_nn,
                       dat_file_name_nn_scaler,
                       dat_file_name_nn_scaler_train,
                       dat_file_name_nn_scaler_test,
                       start_date_test)

    lstm_forecasting(internal_units, look_back, epoch_count, batch_size)

    # XGBoost Dataset generation
    max_depth = 6
    learning_rate = 0.05
    n_estimators = 5000
    gamma = 0.1

    #dataset_generation(dat_file_name,
    #                   dat_file_name_nn,
    #                   dat_file_name_nn_scaler,
    #                   dat_file_name_nn_scaler_train,
    #                   dat_file_name_nn_scaler_test,
    #                   start_date_test)

    xgboost_forecasting(max_depth, learning_rate, n_estimators, gamma)

    f = open(rep_file_name, 'w')
    print("\n")

    # Funzione di stampa delle statistiche aggiornata
    def print_statistics(df_statistic, file=None):
        print(df_statistic.to_string(index=False), file=file)

    if __name__ == '__main__':
        # Altre parti del tuo codice...

        # Apertura del file per scrivere il report
        with open(rep_file_name, 'w') as f:
            print_statistics(df_statistic)
            print_statistics(df_statistic, file=f)


    # Apertura del file per scrivere il report
    with open(rep_file_name, 'w') as f:
        print_statistics(df_statistic)
        print_statistics(df_statistic, file=f)
    f.close()

    print("\n")
    print(df_predictions.head())
    df_predictions.to_csv(prd_file_name)

    f = open(prm_file_name, 'w')

    print(f'look_back: {look_back}\n', file=f)
    print(f'internal_units: {internal_units}\n', file=f)
    print(f'batch_size: {batch_size}\n', file=f)
    print(f'epoch_count: {epoch_count}\n', file=f)

    print(f'max_depth: {max_depth}\n', file=f)
    print(f'learning_rate: {learning_rate}\n', file=f)
    print(f'n_estimators: {n_estimators}\n', file=f)
    print(f'gamma: {gamma}\n', file=f)

    f.close()


    print(f'\nDONE. TOTAL EXECUTION TIME: {(time.time() - start):.3f} sec.')