import pandas as pd

from defines import *
from functions import *
from xml_read import *
from xml_write import *
from tensorflow import keras

from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def data_preparation(data: dict[str, pd.DataFrame], time_step: pd.Timedelta, missing_count_threshold,
                     missing_eat_threshold) -> dict[str, pd.DataFrame]:
    cleaned_data = drop_days_with_missing_glucose_data(data, missing_count_threshold)
    # cleaned_data = drop_days_with_missing_eat_data(cleaned_data, missing_eat_threshold)
    # cleaned_data = fill_glucose_level_data_continuous(cleaned_data, time_step)
    return cleaned_data


def model(train_x, validX, validY, train_y, look_back):
    # input1 = Input(shape=(time_span, 1))
    # x11 = LSTM(units=mem_cells, activation= ‘relu’, return_sequences = False)
    # x12 = x11(input1)
    # x13 = Dense(units=3, activation= ‘relu’)
    # x1 = x13(x12)
    # model = Model(inputs = [input1, input2, input3, input4], outputs=[out1, out2])
    # model.compile(loss = ‘mean_squared_error’, optimizer = keras.optimizers.Adam(0.001))

    input1 = Input(shape=(1,look_back))
    x11 = LSTM(units=100, activation="relu", return_sequences=False)
    x12 = x11(input1)
    x13 = Dense(units=1, activation="relu")
    x1 = x13(x12)
    out2 = Dense(1)(x1)
    glucose_model = Model(inputs=[input1], outputs=[out2])
    glucose_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.001), metrics=["accuracy"])
    history = glucose_model.fit(train_x,train_y, epochs=10, batch_size=look_back, verbose=1, shuffle=False, validation_data=(validX, validY))
    return history, glucose_model


def train_test_valid_split(glucose_data: pd.DataFrame):
    cleaned_data = {}
    for key in glucose_data.keys():
        cleaned_data[key] = glucose_data[key].__deepcopy__()
    cleaned_data = pd.DataFrame(cleaned_data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    cleaned_data = scaler.fit_transform(cleaned_data)
    idx = int(0.8 * int(cleaned_data.shape[0]))
    validIdx = int(0.1 * int(cleaned_data.shape[0]))
    train_x = cleaned_data[:idx]
    valid_x = cleaned_data[idx:idx+validIdx]
    test_x = cleaned_data[idx+validIdx:]
    return train_x, valid_x, test_x

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# ctrl+alt+shift+L REFORMATS CODE

if __name__ == "__main__":
    data, patient_data = load(TRAIN2_540_PATH)
    clean_data = data_preparation(data, pd.Timedelta(5, "m"), 30, 2)
    clean_data["glucose_level"]['ts'] = pd.to_numeric(pd.to_datetime(clean_data["glucose_level"]['ts']))
    train, valid, test = train_test_valid_split(clean_data["glucose_level"])
    print(train.shape)
    look_back = 120
    trainX, trainY = create_dataset(train, look_back)
    validX, validY = create_dataset(valid, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print(train.shape)
    history, model = model(trainX, validX, validY, trainY,  look_back)
    print(history)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = 10
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
