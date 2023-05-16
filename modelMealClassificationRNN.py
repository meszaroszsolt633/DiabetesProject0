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
from keras.layers import Input, LSTM, Conv1D, MaxPooling1D
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import tensorflow_addons as tfa


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

    count_zeros, count_ones = count_ones_and_zeros(features_train_y)
    print("train:\n")
    print(f"Number of 0s: {count_zeros}")
    print(f"Number of 1s: {count_ones}")

    count_zeros, count_ones = count_ones_and_zeros(features_validation_y)
    print("validation:\n")
    print(f"Number of 0s: {count_zeros}")
    print(f"Number of 1s: {count_ones}")

    # Use if data separation is needed, use only one dataframe
    #train, valid = train_test_valid_split(features_train_combined,0.8)
    #trainX, trainY = create_dataset(train, look_back)
    #validX, validY = create_dataset(valid, look_back)




    scaler = MinMaxScaler(feature_range=(0, 1))
    features_train_x = pd.DataFrame(scaler.fit_transform(features_train_x.values), columns=features_train_x.columns,
                                    index=features_train_x.index)
    features_validation_x = pd.DataFrame(scaler.transform(features_validation_x.values),
                                         columns=features_validation_x.columns, index=features_validation_x.index)


    trainY_np = features_train_y.values
    validY_np = features_validation_y.values

    trainX,trainY = create_dataset(features_train_x,trainY_np, lookback)
    validX,validY = create_dataset(features_validation_x,validY_np, lookback)


   #trainX = np.reshape(trainX.shape[0], trainX.shape[1], 1)
   #validX = np.reshape(validX.shape[0], validX.shape[1], 1)

    print("trainX:",trainX.shape)
    print("trainY:",trainY.shape)
    print("validX:",validX.shape)
    print("validY:", validY.shape)


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
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
        #tfa.metrics.F1Score(num_classes=1,average='macro',threshold=0.5)
        ])
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
