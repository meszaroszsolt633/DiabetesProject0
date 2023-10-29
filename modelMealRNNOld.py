import pandas as pd
from defines import *
import numpy as np
from statistics import stdev
from scipy import signal

from functions import create_dataset, train_test_valid_split, data_preprocess
from xml_read import load
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from scipy import ndimage
from keras.models import Model
from keras import metrics
from keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler



def model2(data):
    feature1 = data['glucose_level']
    feature1['carbs'] = ""
    feature1['carbs'] = feature1['carbs'].apply(lambda x: 0)
    feature2 = data['meal']
    feature2 = feature2.drop(['type'], axis=1)

    feature2['carbs'] = feature2['carbs'].apply(lambda x: 1)
    features = pd.concat([feature1, feature2])
    features = features.sort_values(by='ts', ignore_index=True)
    featureY = features['carbs']
    featureX = features['value']
    featureY = ndimage.maximum_filter(featureY, size=10)
    featureY = pd.DataFrame(featureY)
    featureX = pd.DataFrame(featureX)
    featureX = featureX.fillna(method='ffill')
    features2 = pd.concat([featureY, featureX], axis=1)

    features2=features2.astype(float)

    look_back = 60
    train, valid = train_test_valid_split(features2)
    trainX, trainY = create_dataset(train, look_back)
    validX, validY = create_dataset(valid, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
    modelMeal(trainX, validX, validY, trainY, look_back)


def modelMeal(train_x, validX, validY, train_y, look_back):
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
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy',
                       metrics.Precision(name='precision'),
                       metrics.Recall(name='recall')])
    model.fit(train_x, train_y, epochs=50, callbacks=[es_callback, modelckpt_callback], verbose=1, shuffle=False,
              validation_data=(validX, validY))

    prediction = model.predict(validX)
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data, patient_data = load(TRAIN2_540_PATH)
    clean_data = data_preprocess(data, pd.Timedelta(5, "m"), 30, 3)
    model2(clean_data)



