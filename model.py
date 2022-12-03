import pandas as pd

from defines import *
from functions import *
from xml_read import *
from xml_write import *
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential, Model
from keras.layers import Input, LSTM
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import ELU, PReLU, LeakyReLU

def data_preparation(data: dict[str, pd.DataFrame], time_step: pd.Timedelta, missing_count_threshold, missing_eat_threshold)-> dict[str, pd.DataFrame]:
    cleaned_data = drop_days_with_missing_glucose_data(data, missing_count_threshold)
    #cleaned_data = drop_days_with_missing_eat_data(cleaned_data, missing_eat_threshold)
    cleaned_data = fill_glucose_level_data_continuous(cleaned_data, time_step)
    return cleaned_data

def model(train_x, train_y):
    #input1 = Input(shape=(time_span, 1))
    #x11 = LSTM(units=mem_cells, activation= ‘relu’, return_sequences = False)
    #x12 = x11(input1)
    #x13 = Dense(units=3, activation= ‘relu’)
    #x1 = x13(x12)
    #model = Model(inputs = [input1, input2, input3, input4], outputs=[out1, out2])
    #model.compile(loss = ‘mean_squared_error’, optimizer = keras.optimizers.Adam(0.001))


    input1 = Input(shape=(36, 1))
    x11 = LSTM(units=10, activation= "relu", return_sequences = False)
    x12 = x11(input1)
    x13 = Dense(units=3, activation= "relu")
    x1 = x13(x12)
    out2 = Dense(1)(x1)
    glucose_model = Model(inputs=[input1], outputs=[out2])
    glucose_model.compile(loss= "mean_squared_error", optimizer = keras.optimizers.Adam(0.001))
    history = glucose_model.fit(train_x, train_y, epochs = 10,batch_size = 20,validation_split = 0.3,verbose = 1,shuffle = False)

def train_test_split(glucose_date: pd.DataFrame):





#if __name__ == "__main__":
