import pandas as pd
from defines import *
from model import *
from statistics import stdev
from scipy import signal
from xml_read import *
from xml_write import *
from model import *
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from scipy import ndimage
from keras.models import Model
from keras.layers import Input, LSTM, Conv1D, MaxPooling1D, TimeDistributed, Bidirectional
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return



def create_dataset_RNN_1DConv(X,y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i: (i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def train_valid_split(glucose_data: pd.DataFrame,train_ratio):
    cleaned_data = {}
    for key in glucose_data.keys():
        cleaned_data[key] = glucose_data[key].__deepcopy__()
    cleaned_data = pd.DataFrame(cleaned_data)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # cleaned_data = scaler.fit_transform(cleaned_data)
    idx = int(train_ratio * int(cleaned_data.shape[0]))
    train_x = cleaned_data[:idx]
    test_x = cleaned_data[idx:]
    return train_x, test_x


def model_base_RNN_1DConv(dataTrain, dataValidation, lookback=50, maxfiltersize=10, epochnumber=50):


    #TRAIN

    feature_train1 = dataTrain['glucose_level']
    feature_train1['carbs'] = ""
    feature_train1['carbs'] = feature_train1['carbs'].apply(lambda x: 0)

    feature_train2 = dataTrain['meal']
    feature_train2 = feature_train2.drop(['type'], axis=1)
    feature_train2['carbs'] = feature_train2['carbs'].apply(lambda x: 1)

    features_train = pd.concat([feature_train1, feature_train2])
    features_train = features_train.sort_values(by='ts', ignore_index=True)

    features_train_y = features_train['carbs']
    features_train_y = ndimage.maximum_filter(features_train_y, size=maxfiltersize)
    features_train_y = pd.DataFrame(features_train_y)

    features_train_x = features_train['value']
    features_train_x = pd.DataFrame(features_train_x)
    features_train_x = features_train_x.fillna(method='ffill')
    features_train_combined = pd.concat([features_train_y, features_train_x], axis=1)

    #VALIDATION

    feature_validation1 = dataValidation['glucose_level']
    feature_validation1['carbs'] = ""
    feature_validation1['carbs'] = feature_validation1['carbs'].apply(lambda x: 0)

    feature_validation2 = dataValidation['meal']
    feature_validation2 = feature_validation2.drop(['type'], axis=1)
    feature_validation2['carbs'] = feature_validation2['carbs'].apply(lambda x: 1)

    feature_validation = pd.concat([feature_validation1, feature_validation2])
    feature_validation = feature_validation.sort_values(by='ts', ignore_index=True)

    feature_validation_y = feature_validation['carbs']
    feature_validation_y = ndimage.maximum_filter(feature_validation_y, size=maxfiltersize)
    feature_validation_y = pd.DataFrame(feature_validation_y)

    feature_validation_x = feature_validation['value']
    feature_validation_x = pd.DataFrame(feature_validation_x)
    feature_validation_x = feature_validation_x.fillna(method='ffill')
    feature_validation_combined = pd.concat([feature_validation_y, feature_validation_x], axis=1)

    # Use if data separation is needed, use only one dataframe
    #train, valid = train_test_valid_split(features_train_combined,0.8)
    #trainX, trainY = create_dataset(train, look_back)
    #validX, validY = create_dataset(valid, look_back)

    scaler = MinMaxScaler(feature_range=(0, 1))
    features_train_combined = scaler.fit_transform(features_train_combined.values)
    feature_validation_combined = scaler.transform(feature_validation_combined.values)


    trainX, trainY = create_dataset_RNN_1DConv(features_train_combined[:,1],features_train_combined[:,0], lookback)
    validX, validY = create_dataset_RNN_1DConv(feature_validation_combined[:,1],feature_validation_combined[:,0], lookback)

    #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
   #validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
    print("trainX Shape:",trainX.shape)
    print("trainY Shape:",trainY.shape)
    print("validX Shape:",validX.shape)
    print("validY Shape:",validY.shape)
    trainX = np.expand_dims(trainX, axis=-1)
    validX = np.expand_dims(validX, axis=-1)
    print(trainX.shape[1], trainX.shape[2])

    model_meal_RNN_1DCONV(trainX, validX, validY, trainY, epochnumber,0.01)









if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    dataTrain, patient_data = load(TRAIN2_540_PATH)
    dataValidation,patient_data = load(TEST2_540_PATH)
    # clean_data = data_preparation(data, pd.Timedelta(5, "m"), 30, 3)
    model_base_RNN_1DConv(dataTrain, dataValidation)
