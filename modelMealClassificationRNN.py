import pandas as pd
from defines import *
from functions import load_everything, drop_days_with_missing_glucose_data, drop_days_with_missing_eat_data, \
    fill_glucose_level_data_continuous, loadeveryxml
import numpy as np
from statistics import stdev
from scipy import signal

from modelMealClassificationCNN import data_preparation
from xml_read import load
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from scipy import ndimage
from keras.models import Model
from keras import metrics, Sequential
from keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import tensorflow_addons as tfa


def count_ones_and_zeros(array):
    unique_elements, counts = np.unique(array, return_counts=True)
    counts_dict = dict(zip(unique_elements, counts))

    count_ones = counts_dict.get(1, 0)
    count_zeros = counts_dict.get(0, 0)

    return count_zeros, count_ones

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

def train_test_valid_split(glucose_data: pd.DataFrame):
    cleaned_data = {}
    for key in glucose_data.keys():
        cleaned_data[key] = glucose_data[key].__deepcopy__()
    cleaned_data = pd.DataFrame(cleaned_data)
    cleaned_data.columns = cleaned_data.columns.astype(str)
    scaler = MinMaxScaler(feature_range=(0, 1))
    cleaned_data = scaler.fit_transform(cleaned_data)
    idx = int(0.8 * int(cleaned_data.shape[0]))
    train_x = cleaned_data[:idx]
    test_x = cleaned_data[idx:]
    return train_x,  test_x

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

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 2]
        dataX.append(a)
        dataY.append(dataset[i, 0:2])
    return np.array(dataX), np.array(dataY)

def model_base_RNN(dataTrain, dataValidation, lookback=50, maxfiltersize=10, epochnumber=50, modelnumber=1, learning_rate=0.001, oversampling=False):
    # TRAIN

    feature_train1 = dataTrain['glucose_level']
    feature_train1['carbs'] = ""
    feature_train1['eaten'] = feature_train1['carbs'].apply(lambda x: 0)
    feature_train1['carbs'] = feature_train1['carbs'].apply(lambda x: 0)

    feature_train2 = dataTrain['meal']
    feature_train2 = feature_train2.drop(['type'], axis=1)
    feature_train2['eaten'] = feature_train2['carbs'].apply(lambda x: 1)

    features_train = pd.concat([feature_train1, feature_train2])
    features_train = features_train.sort_values(by='ts', ignore_index=True)

    features_train_y = [features_train['eaten'],features_train['carbs']]
    # Apply maximum filter separately to each column
    eaten_max_filtered = ndimage.maximum_filter(features_train_y[:, 0], size=maxfiltersize)
    carbs_max_filtered = ndimage.maximum_filter(features_train_y[:, 1], size=maxfiltersize)

    # Combine the filtered columns back into a single array
    features_train_y = pd.DataFrame(np.column_stack((eaten_max_filtered, carbs_max_filtered)))

    features_train_x = features_train['value']
    features_train_x = pd.DataFrame(features_train_x)
    features_train_x = features_train_x.fillna(method='ffill')
    features_train_x['value'] = features_train_x['value'].astype('float64')

    # VALIDATION

    feature_validation1 = dataValidation['glucose_level']
    feature_validation1['carbs'] = ""
    feature_validation1['eaten'] = feature_validation1['carbs'].apply(lambda x: 0)
    feature_validation1['carbs'] = feature_validation1['carbs'].apply(lambda x: 0)

    feature_validation2 = dataValidation['meal']
    feature_validation2 = feature_validation2.drop(['type'], axis=1)
    feature_validation2['eaten'] = feature_validation2['carbs'].apply(lambda x: 1)

    features_validation = pd.concat([feature_validation1, feature_validation2])
    features_validation = features_validation.sort_values(by='ts', ignore_index=True)

    features_validation_y = [features_validation['eaten'],features_validation['carbs']]
    eaten_max_filtered = ndimage.maximum_filter(features_validation_y[:, 0], size=maxfiltersize)
    carbs_max_filtered = ndimage.maximum_filter(features_validation_y[:, 1], size=maxfiltersize)

    # Combine the filtered columns back into a single array
    features_validation_y = pd.DataFrame(np.column_stack((eaten_max_filtered, carbs_max_filtered)))

    features_validation_x = features_validation['value']
    features_validation_x = pd.DataFrame(features_validation_x)
    features_validation_x = features_validation_x.fillna(method='ffill')
    features_validation_x['value'] = features_validation_x['value'].astype('float64')

    featuresvalidation = pd.concat([features_validation_y, features_validation_x], axis=1)
    featurestrain = pd.concat([features_train_y, features_train_x], axis=1)

    featurestrain.columns = featurestrain.columns.astype(str)
    featuresvalidation.columns = featuresvalidation.columns.astype(str)

    scaler = MinMaxScaler(feature_range=(0, 1))
    featurestrain = scaler.fit_transform(featurestrain)
    featuresvalidation = scaler.transform(featuresvalidation)
    if (oversampling == True):
        trainY = featurestrain[:, 0:2]
        trainX = featurestrain[:, 1]
        trainY = trainY.reshape(-1, 1)
        trainX = trainX.reshape(-1, 1)
        smote = SMOTE(random_state=42)
        trainX, trainY = smote.fit_resample(trainX, trainY)
        featurestrain = np.column_stack((trainY, trainX))
    trainX, trainY = create_dataset(featurestrain, lookback)
    validX, validY = create_dataset(featuresvalidation, lookback)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))

    print("trainX:", trainX.shape)
    print("trainY:", trainY.shape)
    print("validX:", validX.shape)
    print("validY:", validY.shape)

    model_meal_RNN_multioutput(trainX, trainY, validX, validY,  epochnumber, learning_rate)


def model_meal_RNN(train_x, train_y, validX, validY,  epochnumber, lrnng_rate):
    model = keras.Sequential()

    opt = keras.optimizers.Adam(learning_rate=lrnng_rate)
    path_checkpoint = "modelMealRNN_checkpoint.h5"
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=15)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    model.add(LSTM(256, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",
                                                                      tf.keras.metrics.Precision(name="precision"),
                                                                      tf.keras.metrics.Recall(name="recall"),
                                                                      tf.keras.metrics.AUC(name="auc"),
                                                                      tfa.metrics.F1Score(num_classes=1,
                                                                                          average='macro',
                                                                                          threshold=0.5)
                                                                      ])
    history = model.fit(train_x, train_y, epochs=epochnumber, callbacks=[es_callback, reduce_lr, modelckpt_callback],
                        verbose=1, shuffle=False,
                        validation_data=(validX, validY))

    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()

def model_meal_RNN_multioutput(train_x, train_y, validX, validY, epochnumber, lrnng_rate):
    opt = keras.optimizers.Adam(learning_rate=lrnng_rate)
    path_checkpoint = "modelMealRNN_checkpoint.h5"
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=15)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    input1 = Input(shape=(train_x.shape[1], train_x.shape[2]))
    input2 = Input(shape=(train_x.shape[1], train_x.shape[2]))

    # Shared LSTM layers
    lstm1 = LSTM(256, return_sequences=True)(input1)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(128, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)
    lstm3 = LSTM(64, return_sequences=False)(dropout2)
    dropout3 = Dropout(0.3)(lstm3)

    # Output 1
    output1 = Dense(1, activation="sigmoid")(dropout3)

    # Output 2 (conditional output)
    output2 = Dense(1, activation="sigmoid")(dropout3)
    output2 = tf.keras.layers.Multiply()([output1, output2])

    # Create the model
    model = Model(inputs=[input1, input2], outputs=[output1, output2])
    model.summary()

    # Compile and train the model
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",
                                                                  tf.keras.metrics.Precision(name="precision"),
                                                                  tf.keras.metrics.Recall(name="recall"),
                                                                  tf.keras.metrics.AUC(name="auc"),
                                                                  tfa.metrics.F1Score(num_classes=1,
                                                                                      average='macro',
                                                                                      threshold=0.5)
                                                                  ])
    history = model.fit([train_x, train_x], [train_y, train_y], epochs=epochnumber,
                                                                callbacks=[es_callback, reduce_lr, modelckpt_callback],
                                                                verbose=1, shuffle=False,
                                                                validation_data=([validX, validX], [validY, validY]))

    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train, test = loadeveryxml()

    #train,patientdata=load(TRAIN2_540_PATH)
    #test,patientdata=load(TEST2_540_PATH)
    train = data_preparation(train, pd.Timedelta(5, "m"), 30, 3)
    test = data_preparation(test, pd.Timedelta(5, "m"), 30, 3)
    model_base_RNN(dataTrain=train,dataValidation=test,lookback=50,maxfiltersize=10,epochnumber=200,modelnumber=2,learning_rate=0.001,oversampling=False)
