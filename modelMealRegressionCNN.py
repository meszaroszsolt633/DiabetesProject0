import pandas as pd
from defines import *
from functions import load_everything, drop_days_with_missing_glucose_data, drop_days_with_missing_eat_data, \
    fill_glucose_level_data_continuous, loadeveryxml, create_variable_sliding_window_dataset, \
    loadeverycleanedxml, dataPrepareRegression, data_cleaner
import numpy as np
from statistics import stdev
from scipy import signal
from xml_read import load
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from scipy import ndimage
from keras.models import Model
from keras import backend as bcknd
from keras import metrics, Sequential
from keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import tensorflow_addons as tfa
from functions import dataPrepare


def root_mean_squared_error(y_true, y_pred):
    return bcknd.sqrt(bcknd.mean(bcknd.square(y_pred - y_true)))

def modelRegression(dataTrain, dataTest, backward_slidingwindow,forward_slidingwindow, epochnumber=50,modelnumber=1,scaling=True,learning_rate=0.001,oversampling=False,expansion_factor=4,expansion_multiplier=0.8):

    trainX, trainY, validX, validY = dataPrepareRegression(dataTrain,dataTest,backward_slidingwindow,forward_slidingwindow,scaling=scaling,oversampling=oversampling,expansion_factor=expansion_factor,expansion_multiplier=expansion_multiplier)

    print("trainX:",trainX.shape)
    print("trainY:",trainY.shape)
    print("validX:",validX.shape)
    print("validY:", validY.shape)

    if(modelnumber==1):
        modelCNNRegression(trainX, trainY, validX, validY, epochnumber,learning_rate)
    if(modelnumber==2):
        model_meal_RNN_1DCONV_regression(trainX, trainY, validX, validY, epochnumber,learning_rate)


def modelCNNRegression(train_x, train_y, validX, validY, epochnumber, learning_rate=0.001):

    path_checkpoint = "modelMealCNN_checkpoint.h5"
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=30)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation=None))  # Linear activation for regression

    model.compile(loss="mse", optimizer=opt, metrics=["mae", "mse",root_mean_squared_error])  # Using Mean Squared Error for regression

    history = model.fit(train_x, train_y, epochs=epochnumber, callbacks=[es_callback, reduce_lr, modelckpt_callback], verbose=1, shuffle=False, validation_data=(validX, validY))

    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()



def model_meal_RNN_1DCONV_regression(train_x, train_y, validX, validY, epochnumber, learning_rate=0.001):
    model = Sequential()

    opt = keras.optimizers.Adam(learning_rate=learning_rate,clipvalue=0.5)
    path_checkpoint = "modelMeal_checkpoint.h5"
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=40)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    # Conv1D layers
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # LSTM layers
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))

    # Dense output layer for regression
    model.add(Dense(1, activation=None))

    model.summary()
    model.compile(loss="mse", optimizer=opt, metrics=["mae","mse",root_mean_squared_error()])

    history = model.fit(train_x, train_y, epochs=epochnumber, callbacks=[es_callback, reduce_lr, modelckpt_callback], verbose=1, shuffle=False,
                        validation_data=(validX, validY))

    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()

    # plt.plot(history.history['loss'], label='Training loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    # plt.title('Training VS Validation loss')
    # plt.xlabel('No. of Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()




if __name__ == "__main__":
    train, patient_data = load(TRAIN2_540_PATH)
    test, patient_data = load(TEST2_540_PATH)
    #train,test=loadeveryxml()
    # trainX, trainY,testX,testY = dataPrepare(train, test, 3, 15)
    train = data_cleaner(train, pd.Timedelta(5, "m"), 30, 3)
    test = data_cleaner(test, pd.Timedelta(5, "m"), 30, 3)

    modelRegression(dataTrain=train, dataTest=test, backward_slidingwindow=6, forward_slidingwindow=15,
           epochnumber=100, modelnumber=1, learning_rate=0.001, oversampling=False,expansion_factor=4,expansion_multiplier=1,scaling=False)



