# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython, display

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
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
# SKLearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
# Yfinance
get_ipython().system('pip install yfinance')
import yfinance as yf

get_ipython().system('pip install pandas_market_calendars')
import pandas_market_calendars as mcal

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
    config['time_dir'] = os.path.join(config['time_dir'], '')
    
    config['data_dir'] = os.path.join(config['root_dir'], 'data')
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
    
    config['data_dir'] = os.path.join(config['root_dir'], "data")
    config['model_dir'] = os.path.join(config['time_dir'], 'model')
    config['plot_dir'] = os.path.join(config['time_dir'], 'plot')
    config['result_dir'] = os.path.join(config['time_dir'], 'result')

config['input_col'] = ['<Close>', '<Open>', '<High>', '<Low>']
config['output_col'] = ['<Close>']
config['time_col'] = ['<DTYYYYMMDD>']
# Number of session to prediction as one time
config['prediction_size'] = 1
# For each time model is train, the first is display
config['sample_display_test_size'] = 5
# windows size
config['windows_size'] = 5
config['train_split'] = 0.7
config['validation_split'] = 0.1
config['test_split'] = 0.2
# model config
config['lstm_neuron_count'] = 128
config['lstm_layer_count'] = 5
config['drop_rate'] = 0.2
config['stateful'] = False

# data normalize
config['scaler_feature_range'] = (0, 1)

# train
config['epochs'] = 200
config['batch_size'] = 5
config['start_time'] = datetime(2006, 1, 1, 0, 0)
config['end_time'] = datetime(2016, 11, 13, 0, 0) 
config['force_train'] = False

pd.options.display.max_columns = 12
pd.options.display.max_rows = 24

# disable warnings in Anaconda
warnings.filterwarnings('ignore')

# endregion


# %%
# region Data Loading
def get_data(config, stock_file_name = '000002.SS'):
    data_dir = config['data_dir']
    start_time = config['start_time']
    end_time = config['end_time']
    time_col = config['time_col']
    time_col = time_col[0]
    data_file_path = f'{data_dir}/{stock_file_name}.csv'

    if os.path.exists(data_file_path):
        df_org = pd.read_csv(data_file_path, parse_dates=[time_col])
        # df_org = df_org[np.logical_and(df_org[time_col].dt.to_pydatetime() >= config['start_time'], df_org[time_col].dt.to_pydatetime() <= config['end_time'])]
    else:
        df_org = yf.download(stock_file_name, interval="1d")
        df_org.to_csv(data_file_path)


    df_org = df_org.sort_values(time_col)
    df_org.reset_index(inplace=True)

    return df_org

def calculate_change(df, target_col_name = 'Close', change_col_name = 'Change'):
    df_change = df[target_col_name].copy()
    df_change = df_change.pct_change(periods=1, fill_method='ffill')
    df_change = df_change.fillna(0)

    df[change_col_name] = df_change

    return df

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
    return np.intersect1d(df.columns.values, col_list, assume_unique=True)

# endregion


# %%
# region Declare model
# declare model
def none_to_default(value, value_if_fall):
    try:
        return value_if_fall if value is None else value
    except:
        return value_if_fall

def softmax_axis1(x):
    return softmax(x, axis=1)


def get_model(config = config):
    input_dim = config['input_dim']
    windows_size = config['windows_size']
    output_dim = config['output_dim']
    lstm_neuron_count = none_to_default(config['lstm_neuron_count'], 128)
    lstm_layer_count = none_to_default(config['lstm_layer_count'], 5)
    drop_rate = none_to_default(config['drop_rate'], 0.2)
    stateful = none_to_default(config['stateful'], False)
    batch_size = config['batch_size']
    model = Sequential()
    
    if stateful:
      model.add(LSTM(units=lstm_neuron_count, batch_input_shape=(batch_size, windows_size, input_dim), return_sequences=True, stateful = stateful, dropout=drop_rate, recurrent_dropout=drop_rate))
    else:
      model.add(LSTM(units=lstm_neuron_count, input_shape=(windows_size, input_dim), return_sequences=True, stateful = stateful, dropout=drop_rate, recurrent_dropout=drop_rate))

    for i in range(lstm_layer_count - 2):
        model.add(LSTM(units=lstm_neuron_count, return_sequences=True, stateful = stateful, dropout=drop_rate, recurrent_dropout=drop_rate))
        model.add(Dropout(rate=drop_rate))
    
    model.add(LSTM(units=lstm_neuron_count, return_sequences=False, stateful = stateful, dropout=drop_rate, recurrent_dropout=drop_rate))
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(output_dim, activation='linear'))
    opt = optimizers.Adam(lr=0.05, beta_1=0.99, beta_2=0.999)
    softmax_activation = softmax_axis1
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
        EarlyStopping(monitor='loss', patience=100),
        ModelCheckpoint(filepath=model_save_fname, monitor='loss', save_best_only=True)
    ]
    epochs = none_to_default(config['epochs'], 1000)
    batch_size = none_to_default(config['batch_size'], 1000)
    train_split = none_to_default(config['train_split'], 0.7)
    validation_split = none_to_default(config['validation_split'], 0.1)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split= float(validation_split) / float(train_split),
        verbose=1,
        callbacks=callbacks,
        shuffle=False)

    model.save(model_save_fname)
    
    return history

def load_save_model(stock_name):
    model_save_fname = os.path.join(config['model_dir'], '%s.h5' % (stock_name))
    scaler_save_fname = os.path.join(config['model_dir'], '%s.scaler' % (stock_name))
    
    if os.path.exists(model_save_fname) and os.path.exists(scaler_save_fname):
        return {'model' : load_model(model_save_fname), 'scaler': joblib.load(scaler_save_fname)}
        
    return None

# endregion
# %%
def plot_test_result(df_test_result, stock_name, config):
    # Plotly
    output_col = config['output_col']
    prediction_col = config['prediction_col']
    time_col = config['time_col']
    trace0 = go.Scatter(
        x=df_test_result.index,
        y=df_test_result[output_col[0]],
        name='Thực tế',
        line=dict(
            color=('#5042f4'),
            width=2)
    )

    trace1 = go.Scatter(
        x=df_test_result.index,
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
    plot_dir = config['plot_dir']
    fig.show()
    fig.write_html(os.path.join(plot_dir, '%s_test.html' % (stock_name)), auto_open=False)
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
    config['time_col'] = time_col
    config['input_dim'] = len(input_col)
    config['output_dim'] = len(output_col)
    

    df_train, df_test = train_test_split(df, test_size=test_split, shuffle=False)
    
    model = get_model(config=config)
    
    start = timer()

    # Handle data
    scaler_feature_range = config.get('scaler_feature_range', (0, 1))
    scaler = MinMaxScaler(feature_range=scaler_feature_range)
    scaled_cols = scaler.fit(df_train[input_col])

    # Save scaler
    scaler_save_fname = os.path.join(config['model_dir'], '%s.scaler' % (stock_name))
    joblib.dump(scaler, scaler_save_fname) 
    
    # Transform train data
    scaled_cols = scaler.transform(df_train[input_col])
    df_train[input_col] = scaled_cols

    X_train, y_train, time_train = preprocessing_data(df_train, config)

    # Reshape data
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))

    # Perform n_train
    history = train_model(model, X_train, y_train, stock_name)
    
    run_time = timer() - start

    return {'scaler' : scaler, 'model' : model, 'history' : history, 'run_time' : run_time} 
    # %%
def do_test(stock_name, data, config = config):

    prediction_size = config['prediction_size']
    input_col =  config['input_col']
    output_col =  config['output_col']
    time_col =  config['time_col']
    batch_size =  config['batch_size']
    
    df = get_data(config, stock_name)
    input_col = get_df_intersect_col(df, input_col)
    output_col = get_df_intersect_col(df, output_col)
    output_col = output_col[0]
    time_col = get_df_intersect_col(df, time_col)
    time_col =  time_col[0]
    df_org = df[[output_col, time_col]].copy()

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

    df_test_result = pd.DataFrame(time_test, columns=[time_col])
    prediction_col = f'{output_col} Prediction'
    config['prediction_col'] = prediction_col
    df_test_result[config['prediction_col']] = y_pred
    df_test_result.set_index(time_col, inplace=True)

    df_test_result = df_test_result.join(df_org.set_index(time_col))


    score = model.evaluate(X_test, y_test, batch_size, 1)
    mae = mean_absolute_error(df_test_result[output_col], df_test_result[prediction_col])
    mse = mean_squared_error(df_test_result[output_col], df_test_result[prediction_col])
    mape = mean_absolute_percentage_error(df_test_result[output_col], df_test_result[prediction_col])
    rrmse = relative_root_mean_square_error(df_test_result[output_col], df_test_result[prediction_col])

    # File to save first results\n
    result_dir = config['result_dir']
    result_save_fname = os.path.join(result_dir, 'result_%s.csv' % (stock_name))
    of_connection = open(result_save_fname, 'w')
    writer = csv.writer(of_connection)
    # Write the headers to the file\n
    writer.writerow(['stock_name', 'score', 'mae', 'mse', 'mape', 'rrmse', 'time_stamp'])
    writer.writerow([stock_name, score, mae, mse, mape, rrmse, datetime.now().strftime('%d%m%Y_%H%M%S')])
    of_connection.close()
    # write data
    return  {'score' : score, 'mae' : mae, 'df': df_test_result, 'mse' : mse, 'mape' : mape, 'rrmse' : rrmse}

def make_future_prediction(model, scaler, future_step, config):
    df = get_data(config, stock_name)
    windows_size = config['windows_size']
    input_col = config['input_col']
    output_col = config['output_col']
    time_col = config['time_col']
    prediction_col = config['prediction_col']

    time_col = time_col[0]
    prediction_size = config['prediction_size']
    batch_size = config['batch_size']

    stock_calendar = mcal.get_calendar('stock')
    time = df[time_col][-1:].values[0]
    stock_time = stock_calendar.valid_days(start_date=time + np.timedelta64(1, 'D'), end_date=time + np.timedelta64(future_step * 2, 'D'))
    
    pred_res = df[input_col][-batch_size:].copy()
    pred_res[prediction_col] = pred_res[output_col]
    '''Generates the next data window from the given index location i'''
    for step in range(future_step):
        x = pred_res[input_col][-windows_size:].values
        x = scaler.transform(x)
        x = x.reshape(1, x.shape[0], x.shape[1])
            
        y_pred = model.predict(x)
        y_pred = np.repeat(y_pred, len(input_col), axis=1)
        y_pred = scaler.inverse_transform(y_pred)[:, [0]][0][0]

        pred_res = pred_res.append({time_col : stock_time[step], prediction_col:y_pred, output_col:np.repeat(y_pred, len(input_col), axis=1)}, ignore_index=True )

    return pred_res

def plot_furure_prediction(df, df_predict, stock_name, config):
    # Plotly
    output_col = config['output_col']
    prediction_col = config['prediction_col']
    time_col = config['time_col']
    time_col = time_col[0]
    trace0 = go.Scatter(
        x=df.index,
        y=df[output_col[0]],
        name='Thực tế',
        line=dict(
            color=('#5042f4'),
            width=2)
    )

    trace1 = go.Scatter(
        x=df_predict[time_col],
        y=df_predict[prediction_col],
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
    plot_dir = config["plot_dir"]
    fig.write_html(os.path.join(plot_dir, '%s_predict.html' % (stock_name)), auto_open=False)
    fig.show()
# endregion

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
    stock_name_list = ['FLC']

    for stock_name in stock_name_list:
        force_train = config.get('force_train', False)
        train_result = load_model(stock_name, config)
        if train_result is None or force_train:
            train_result = do_train(stock_name, config)
        test_result = do_test(stock_name, train_result ,config)
        
        future_predict = make_future_prediction(train_result['model'], train_result['scaler'] ,10, config)
        plot_test_result(test_result['df'], stock_name, config)
        plot_furure_prediction(test_result['df'], future_predict, stock_name, config)

        result_dir = config['result_dir']
        future_pred_file_path = f'{result_dir}/{stock_name}_pred.csv'
        future_predict.to_csv(future_pred_file_path)

        
