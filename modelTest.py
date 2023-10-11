import pandas as pd
from defines import *
import numpy as np
from statistics import stdev
from scipy import signal

from functions import count_ones_and_zeros, create_dataset, data_preparation
from xml_read import load
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from scipy import ndimage
from keras.models import Model
from keras import metrics
from keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE



def model2(dataTrain, dataValidation, lookback=50, maxfiltersize=10, epochnumber=50):


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
    count_zeros, count_ones = count_ones_and_zeros(features_train_y)
    print("train:\n")
    print(f"Number of 0s: {count_zeros}")
    print(f"Number of 1s: {count_ones}")
    features_train_y = ndimage.maximum_filter(features_train_y, size=maxfiltersize)
    features_train_y = pd.DataFrame(features_train_y)

    features_train_x = features_train['value']
    features_train_x = pd.DataFrame(features_train_x)
    features_train_x = features_train_x.fillna(method='ffill')
    features_train_x['value'] = features_train_x['value'].astype('float64')


    #VALIDATION

    feature_validation1 = dataValidation['glucose_level']
    feature_validation1['carbs'] = ""
    feature_validation1['carbs'] = feature_validation1['carbs'].apply(lambda x: 0)

    feature_validation2 = dataValidation['meal']
    feature_validation2 = feature_validation2.drop(['type'], axis=1)
    feature_validation2['carbs'] = feature_validation2['carbs'].apply(lambda x: 1)

    features_validation = pd.concat([feature_validation1, feature_validation2])
    features_validation = features_validation.sort_values(by='ts', ignore_index=True)

    features_validation_y = features_validation['carbs']
    count_zeros, count_ones = count_ones_and_zeros(features_validation_y)
    print("validation:\n")
    print(f"Number of 0s: {count_zeros}")
    print(f"Number of 1s: {count_ones}")
    features_validation_y = ndimage.maximum_filter(features_validation_y, size=maxfiltersize)
    features_validation_y = pd.DataFrame(features_validation_y)

    features_validation_x = features_validation['value']
    features_validation_x = pd.DataFrame(features_validation_x)
    features_validation_x = features_validation_x.fillna(method='ffill')
    features_validation_x['value'] = features_validation_x['value'].astype('float64')


    featuresvalidation  = pd.concat([features_validation_y, features_validation_x], axis=1)
    featurestrain=pd.concat([features_train_y,features_train_x],axis=1)

    featurestrain.columns = featurestrain.columns.astype(str)
    featuresvalidation.columns = featuresvalidation.columns.astype(str)



    scaler = MinMaxScaler(feature_range=(0, 1))
    featurestrain=scaler.fit_transform(featurestrain)
    featuresvalidation=scaler.transform(featuresvalidation)


    trainY_np = features_train_y.values
    validY_np = features_validation_y.values

    trainX,trainY = create_dataset(featurestrain, lookback)
    validX,validY = create_dataset(featuresvalidation, lookback)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))

    print("trainX:",trainX.shape)
    print("trainY:",trainY.shape)
    print("validX:",validX.shape)
    print("validY:", validY.shape)


    modelMeal(trainX, validX, validY, trainY, epochnumber)


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
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy',
                       metrics.Precision(name='precision'),
                       metrics.Recall(name='recall')])
    model.fit(train_x, train_y, epochs=100, callbacks=[ modelckpt_callback], verbose=1, shuffle=False,
              validation_data=(validX, validY))

    prediction = model.predict(validX)
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataTrain, patient_data = load(TRAIN2_544_PATH)
    dataValidation, patient_data = load(TEST2_544_PATH)
    dataTrain = data_preparation(dataTrain, pd.Timedelta(5, "m"), 30, 3)
    dataValidation = data_preparation(dataValidation, pd.Timedelta(5, "m"), 30, 3)
    model2(dataTrain,dataValidation)



