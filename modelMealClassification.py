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
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #cleaned_data = scaler.fit_transform(cleaned_data)
    idx = int(0.8 * int(cleaned_data.shape[0]))
    train_x = cleaned_data[:idx]
    test_x = cleaned_data[idx:]
    return train_x,  test_x


def model2(data, windowsize):
    feature1=data['glucose_level']
    feature1['carbs']=""
    feature1['carbs']=feature1['carbs'].apply(lambda x:0)
    feature2=data['meal']
    feature2=feature2.drop(['type'],axis=1)

    feature2['carbs'] = feature2['carbs'].apply(lambda x: 1)
    features = pd.concat([feature1, feature2])
    features = features.sort_values(by='ts', ignore_index=True)
    featureY=features['carbs']
    featureX=features['value']
    featureY = ndimage.maximum_filter(featureY, size=10)
    featureY=pd.DataFrame(featureY)
    featureX=pd.DataFrame(featureX)
    featureX=featureX.fillna(method='ffill')
    features2=pd.concat([featureY,featureX],axis=1)

    look_back=windowsize
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
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=1000,callbacks=[es_callback, modelckpt_callback],verbose=1, shuffle=False,validation_data=(validX,validY))

    prediction = model.predict(validX)
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data, patient_data = load(TRAIN2_540_PATH)
    #clean_data = data_preparation(data, pd.Timedelta(5, "m"), 30, 3)
    model2(data,30)