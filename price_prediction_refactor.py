# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/vyphamhung10/khoa_luan/blob/master/price_prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# -*- coding: utf-8 -*-
# region Import
# Data download
# Import basic
import csv
import math
import os
import warnings
# Init google drive
# from google.colab import drive
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import pandas as pd
# Plottool
import plotly.graph_objs as go
# IPython
from IPython.display import display
# Hyperopt bayesian optimization
from hyperopt import hp, Trials, tpe, fmin, STATUS_OK, partial
# Keras
import tensorflow as tf
import tensorflow 
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
from tensorflow.keras.initializers import random_normal, Ones 
from tensorflow.keras.layers import LSTM, Dropout, Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
# SKLearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Yfinance
get_ipython().system('pip install yfinance')
import yfinance as yf

# endregion


# %%
# Region config config
try:
  from google.colab import drive
  IN_COLAB = True
except:
  IN_COLAB = False

config = {}
config['current_timestamp'] = datetime.now().strftime('%d%m%Y_%H%M%S')
if not IN_COLAB:
    
    # region File mount and config
    # drive.mount('/content/gdrive', force_remount=True)
    config['root_dir'] = ""
    
    config['time_dir'] = os.path.join(config['root_dir'], "result")
    config['time_dir'] = os.path.join(config['time_dir'], current_timestamp)
    
    config['data_dir'] = root_dir + 'data'
    config['model_dir'] = os.path.join(config['time_dir'], 'model')
    config['plot_dir'] = os.path.join(config['time_dir'], 'plot')
    config['result_dir'] = os.path.join(config['time_dir'], 'result')
    # Create folder if not exists
    
    if not os.path.exists(config['data_dir']):
        os.makedirs(config['data_dir'])

    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])
    
    if not os.path.exists(config['plot_dir']):
        os.makedirs(config['plot_dir'])
        
    if not os.path.exists(config['result_dir']):
        os.makedirs(config['result_dir'])
else:
    drive.mount('/content/gdrive', force_remount=True)
    config['root_dir'] = "/content/gdrive/My Drive/stock"
    
    config['time_dir'] = os.path.join(config['root_dir'], "result")
    
    config['data_dir'] = os.path.join(root_dir, "data")
    config['model_dir'] = os.path.join(config['time_dir'], 'model')
    config['plot_dir'] = os.path.join(config['time_dir'], 'plot')
    config['result_dir'] = os.path.join(config['time_dir'], 'result')

config['input_col'] = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Change']
config['output_col'] = ['Close']
config['time_col'] = ['Date']
# Number of session to prediction as one time
config['prediction_size'] = 1
# For each time model is train, the first is display
config['sample_display_test_size'] = 5
# windows size
config['windows_size'] = 7
config['train_split'] = 0.7
config['validation_split'] = 0.1
config['test_split'] = 0.1


pd.options.display.max_columns = 12
pd.options.display.max_rows = 24

# disable warnings in Anaconda
warnings.filterwarnings('ignore')

# endregion


# %%
# region Data Loading
def get_data(config, stock_file_name = '000002.SS'):
    data_dir = config['data_dir']
    data_file_path = f'{data_dir}/{stock_file_name}.csv'

    if os.path.exists(data_file_path):
        df_org = pd.read_csv(data_file_path, parse_dates=['Date'])
    else:
        df_org = yf.download(stock_file_name, start="2006-01-01", end="2016-10-19", interval="1d")
        df_org.to_csv(data_file_path)


    df_org = df_org.sort_values('Date')
    df_org.reset_index(inplace=True)

def calculate_change(df, target_col_name = 'Close', change_col_name = 'Change'):
    df_change = df[target_col_name].copy()
    df_change = df_change.pct_change(periods=1, fill_method='ffill')
    df_change = df_change.fillna(0)

    df[change_col_name] = df_change

# region Data ploting
def plot_ohlc(df, stock_name):
    trace = go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    increasing=dict(line=dict(color='#58FA58')),
                    decreasing=dict(line=dict(color='#FA5858')))

    layout = {
        'title': f'{stock_name} Historical Price',
        'xaxis': {'title': 'Date',
                  'rangeslider': {'visible': False}},
        'yaxis': {'title': f'Price'}
    }

    data = [trace]

    fig = go.Figure(data=data, layout=layout)
    fig.show()
    return fig

def get_df_intersect_col(df, col_list):
    return np.intersect1d(df.columns.values, columns, assume_unique=True)

# endregion


# %%
# region Declare model
# declare model
def none_to_default(value, value_if_fall):
    try:
        return value_if_fall if value is None else value
    except:
        return value_if_fall


def get_model(config = config):
    input_dim = config['input_dim']
    window_size = config['window_size']
    output_dim = config['output_dim']
    lstm_neuron_count = none_to_default(config['lstm_neuron_count'], 128)
    lstm_layer_count = none_to_default(config['lstm_layer_count'], 5)
    drop_rate = none_to_default(config['drop_rate'], 0.2)
    stateful = none_to_default(config['stateful'], False)
    model = Sequential()

    model.add(LSTM(units=lstm_neuron_count, input_shape=(window_size, input_dim), return_sequences=True, stateful = stateful))
    model.add(Dropout(rate=drop_rate))

    for i in range(lstm_layer_count - 2):
        model.add(LSTM(units=lstm_neuron_count, return_sequences=True))
        model.add(Dropout(rate=drop_rate))
    
    model.add(LSTM(units=lstm_neuron_count, return_sequences=False))
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(output_dim, activation='linear'))
    opt = optimizers.Adam(lr=0.05, beta_1=0.99, beta_2=0.999)
    softmax_activation = softmax(x, axis=1)
    model.compile(loss='MSE', optimizer='adam')
    
    return model


# endregion

# region Error metric
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_square_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean((y_true - y_pred) / y_true)


def relative_root_mean_square_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    res = (y_true - y_pred) / y_true
    res = np.power(res, 2)
    res = np.mean(res)
    res = math.sqrt(res)

    return res


# endregion

# region Data preprocessing
# reprocessing data
def next_window(df, i, config = config):
    windows_size = config['windows_size']
    prediction_size = config['prediction_size']
    input_col = config['input_col']
    output_col = config['output_col']
    time_col = config['time_col']

    '''Generates the next data window from the given index location i'''
    window = df[i: i + windows_size + prediction_size]
    x = window[input_col][:-prediction_size]
    y = window[output_col][-prediction_size:]
    y_time = window[time_col][-prediction_size:]
    return x, y, y_time

def smooting_data(df, config = config):
    windows_size = config['windows_size']
    return df.ewm(span=windows_size).mean()

def preprocessing_data(df, config = config):
    '''
    Create x, y train data windows
    Warning: batch method, not generative, make sure you have enough memory 
    '''
    windows_size = config['windows_size']
    prediction_size = config['prediction_size']
    input_col = config['input_col']
    output_col = config['output_col']
    time_col = config['time_col']

    data_x = []
    data_y = []
    data_y_time = []
    for i in range(len(df) - windows_size - prediction_size):
        x, y, y_time = next_window(df, i, config)
        data_x.append(x.values)
        data_y.append(y.values)
        data_y_time.append(y_time)

    time = pd.concat(data_y_time)

    return np.array(data_x), np.array(data_y), time.values

# endregion


# %%
# region Model train
# Trainning model
def train_model(model, X_train, y_train, save_fname):
    model_save_fname = os.path.join(config['model_dir'], '%s.h5' % (save_fname))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20),
        ModelCheckpoint(filepath=model_save_fname, monitor='val_loss', save_best_only=True)
    ]
    epochs = none_to_default(config['epochs'], 1000)
    batch_size = none_to_default(config['batch_size'], 1000)
    train_split = none_to_default(config['validation_split'], 0.7)
    validation_split = none_to_default(config['validation_split'], 0.1)
    test_split = none_to_default(config['validation_split'], 1.0 / 7)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split= float(train_split) / float(validation_split),
        verbose=1,
        callbacks=callbacks,
        shuffle=False)

    model.save(model_save_fname)
    
    return history


# endregion
# %%
def plot_test_result(df_test_result, stock_name, config):
    # Plotly
    output_col = config['output_col']
    prediction_col = config['prediction_col']
    time_col = config['time_col']
    trace0 = go.Scatter(
        x=df_test_result[time_col[0]],
        y=df_test_result[output_col[0]],
        name='Thực tế',
        line=dict(
            color=('#5042f4'),
            width=2)
    )

    trace1 = go.Scatter(
        x=df_test_result[time_col[0]],
        y=df_test_result[prediction_col],
        name='Dự đoán',
        line=dict(
            color=('#005b4e'),
            width=2,
            dash='dot'
        )  # dash options include 'dash', 'dot', and 'dashdot'
    )

    data = [trace0, trace1]

    # Edit the layout
    layout = dict(title='Biểu đồ dự đoán',
                  xaxis=dict(title='Date'),
                  yaxis=dict(title='Price'),
                  paper_bgcolor='#FFF9F5',
                  plot_bgcolor='#FFF9F5'
                  )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
# endregion

# Region do main thing
def do_train(stock_name, config = config): 
    result = {}

    train_split = config['train_split']
    validation_split = config['validation_split']
    test_split = config['test_split']

    df = get_data(config, stock_name)
    input_col = get_df_intersect_col(df, config['input_col'])
    output_col = get_df_intersect_col(df, config['output_col'])
    time_col = get_df_intersect_col(df, config['time_col'])
    
    config['input_col'] = input_col
    config['output_col'] = output_col
    config['time_col'] = input_col

    df_train, df_test = train_test_split(df, test_size=test_split, shuffle=False)
    
    model = get_model(config=config)
    
    start = timer()

    # Handle data
    
    scaler_feature_range = none_to_default(config['scaler_feature_range'], (0, 1))
    scaler = MinMaxScaler(feature_range=scaler_feature_range)
    scaled_cols = scaler.fit(df_train[input_col])
    scaled_cols = scaler.transform(df_train[input_col])
    df_train[input_col] = scaled_cols
    result['scaler'] = scaler

    X_train, y_train, time_train = preprocessing_data(df_train, windows_size, prediction_size, input_col, output_col, time_col)


    # Reshape data
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))

    # Perform n_train
    history = train_model(model, X_train, y_train, stock_name)

    run_time = timer() - start
    result['run_time'] = run_time

    return result 
    # %%
def do_test(stock_name, data, config = config):

    prediction_size = config['prediction_size']
    input_col =  config['input_col']
    output_col =  config['output_col']
    time_col =  config['time_col']

    df = get_data(config, stock_name)
    input_col = get_df_intersect_col(df, input_col)
    output_col = get_df_intersect_col(df, output_col)
    output_col = output_col[0]
    time_col = get_df_intersect_col(df, time_col)

    test_split = config['test_split']
    df_train, df_test = train_test_split(df, test_size=test_split, shuffle=False)
    scaler = data['scaler']
    scaled_cols = scaler.transform(df_test[input_col])
    df_test[input_col] = scaled_cols

    X_test, y_test, time_test = preprocessing_data(df_test, config)
    
    # Reshape data
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))

    # Test generated loss
    model = data['model']
    y_pred = model.predict(X_test)
    y_pred = np.repeat(y_pred, len(input_col), axis=1)
    y_pred = scaler.inverse_transform(y_pred)[:, [0]]
    y_pred = pd.Series(y_pred.flatten())

    df_test_result = pd.DataFrame(time_test, columns=time_col)
    prediction_col = f'{output_col} Prediction'
    config['prediction_col'] = prediction_col
    df_test_result[config['prediction_col']] = y_pred
    df_test_result.set_index(time_col[0], inplace=True)

    df_test_result = df_test_result.join(df_test)


    score = model.evaluate(X_test, y_test, 10000, 1)
    mae = mean_absolute_error(df_test_result[output_col], df_test_result[prediction_col])
    mse = mean_squared_error(df_test_result[output_col], df_test_result[prediction_col])
    mape = mean_absolute_percentage_error(df_test_result[output_col], df_test_result[prediction_col])
    rrmse = relative_root_mean_square_error(df_test_result[output_col], df_test_result[prediction_col])

    # write data
    return  {'score' : score, 'mae' : mae, 'df': df_test_result, 'mse' : mse, 'mape' : mape, 'rrmse' : rrmse}

def make_future_prediction(model, config, time):
    return True
# %%
# Make future frame For 6 year, 3 year, 1 year, 1 month.

# Hyperparameter Tuning
#   + Train / test split valdiation
#   + Droprate
#   + Activation
#   + Number of layer

# Agents 
# Stock List
if __name__ == "__main__":
    train_result = do_train('sinotec', config)
    test_result = do_test('sinotec', train_result ,config)
    plot_test_result(test_result['df'], 'sinotec')
