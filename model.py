import pandas as pd

from defines import *
from functions import *
from xml_read import *
from xml_write import *
from tensorflow import keras
from modelMealClassification import *
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler


def data_preparation(data: dict[str, pd.DataFrame], time_step: pd.Timedelta, missing_count_threshold,
                     missing_eat_threshold) -> dict[str, pd.DataFrame]:
    cleaned_data = drop_days_with_missing_glucose_data(data, missing_count_threshold)
    cleaned_data = drop_days_with_missing_eat_data(cleaned_data, missing_eat_threshold)
    for key in cleaned_data.keys():
        cleaned_data[key] = cleaned_data[key].reset_index(drop=True)
    cleaned_data = fill_glucose_level_data_continuous(cleaned_data, time_step)
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
    x11 = LSTM(units=256, activation="relu", return_sequences=False)
    x31 = Dense(units=32, activation="relu")
    y21 = Dropout(0.1)
    x41 = Dense(units=12, activation="relu")

    x12 = x11(input1)

    x32=x31(x12)
    y22=y21(x32)


    x1 = x41(y22)

    out2 = Dense(1)(x1)
    glucose_model = Model(inputs=[input1], outputs=[out2])
    glucose_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.001))

    path_checkpoint = "glucoselevelmodel_checkpoint.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = glucose_model.fit(train_x,train_y,callbacks=[es_callback, modelckpt_callback], epochs=1000, batch_size=look_back, verbose=1, shuffle=False, validation_data=(validX, validY))
    visualize_loss(history,"Training and Validation loss")
    return history, glucose_model


def train_test_valid_split(glucose_data: pd.DataFrame):
    cleaned_data = {}
    for key in glucose_data.keys():
        cleaned_data[key] = glucose_data[key].__deepcopy__()
    cleaned_data = pd.DataFrame(cleaned_data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    cleaned_data = scaler.fit_transform(cleaned_data)
    idx = int(0.8 * int(cleaned_data.shape[0]))
    train_x = cleaned_data[:idx]
    test_x = cleaned_data[idx:]
    return train_x,  test_x

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
    train, valid = train_test_valid_split(clean_data["glucose_level"])
    print(train.shape)
    look_back = 120
    trainX, trainY = create_dataset(train, look_back)
    validX, validY = create_dataset(valid, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
    print(train.shape)
    history, model = model(trainX, validX, validY, trainY,  look_back)
    print(history)
