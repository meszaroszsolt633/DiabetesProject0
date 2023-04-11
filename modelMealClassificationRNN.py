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
from keras.layers import Input, LSTM
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


def train_valid_split(glucose_data: pd.DataFrame):
    cleaned_data = {}
    for key in glucose_data.keys():
        cleaned_data[key] = glucose_data[key].__deepcopy__()
    cleaned_data = pd.DataFrame(cleaned_data)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # cleaned_data = scaler.fit_transform(cleaned_data)
    idx = int(0.8 * int(cleaned_data.shape[0]))
    train_x = cleaned_data[:idx]
    test_x = cleaned_data[idx:]
    return train_x, test_x


def model_base_RNN(dataTrain, dataValidation, lookback=50, maxfiltersize=10, epochnumber=50):


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
    #train, valid = train_test_valid_split(features_train_combined)
    #trainX, trainY = create_dataset(train, look_back)
    #validX, validY = create_dataset(valid, look_back)

    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_validation_combined = scaler.fit_transform(feature_validation_combined)
    features_train_combined = scaler.fit_transform(features_train_combined)

    trainX, trainY = create_dataset(features_train_combined, lookback)
    validX, validY = create_dataset(feature_validation_combined, lookback)


    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))

    model_meal_RNN(trainX, validX, validY, trainY, epochnumber)


def model_meal_RNN(train_x, validX, validY, train_y, epochnumber):
    model = keras.Sequential()

    opt = keras.optimizers.Adam(learning_rate=0.01)
    path_checkpoint = "modelMeal_checkpoint.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    model.add(LSTM(128, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(train_x, train_y, epochs=epochnumber, callbacks=[modelckpt_callback], verbose=1, shuffle=False,
              validation_data=(validX, validY))

    prediction = model.predict(validX)
    #Prediction and actual data plot
   #plt.figure(figsize=(20, 6))
   #plt.plot(prediction[0:1440 * 3], label='prediction')
   #plt.plot(validY[0:1440 * 3], label='test_data')
   #plt.legend()
   #plt.show()
    #Loss and validation loss plot
    plt.plot(history.history['loss'],  label='Training loss')
    plt.plot(history.history['val_loss'],  label='Validation loss')
    plt.title('Training VS Validation loss')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    dataTrain, patient_data = load(TRAIN2_540_PATH)
    dataValidation,patient_data = load(TEST2_540_PATH)
    # clean_data = data_preparation(data, pd.Timedelta(5, "m"), 30, 3)
    model_base_RNN(dataTrain, dataValidation)
