import pandas as pd

from defines import *
from functions import *
from xml_read import *
from xml_write import *
from tensorflow import keras

from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


def data_preparation(data: dict[str, pd.DataFrame], time_step: pd.Timedelta, missing_count_threshold,
                     missing_eat_threshold) -> dict[str, pd.DataFrame]:
    cleaned_data = drop_days_with_missing_glucose_data(data, missing_count_threshold)
    # cleaned_data = drop_days_with_missing_eat_data(cleaned_data, missing_eat_threshold)
    # cleaned_data = fill_glucose_level_data_continuous(cleaned_data, time_step)
    return cleaned_data


def model(train_x):
    # input1 = Input(shape=(time_span, 1))
    # x11 = LSTM(units=mem_cells, activation= ‘relu’, return_sequences = False)
    # x12 = x11(input1)
    # x13 = Dense(units=3, activation= ‘relu’)
    # x1 = x13(x12)
    # model = Model(inputs = [input1, input2, input3, input4], outputs=[out1, out2])
    # model.compile(loss = ‘mean_squared_error’, optimizer = keras.optimizers.Adam(0.001))

    input1 = Input(shape=(36, 1))
    x11 = LSTM(units=10, activation="relu", return_sequences=False)
    x12 = x11(input1)
    x13 = Dense(units=3, activation="relu")
    x1 = x13(x12)
    out2 = Dense(1)(x1)
    glucose_model = Model(inputs=[input1], outputs=[out2])
    glucose_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.001))
    history = glucose_model.fit(train_x, epochs=10, batch_size=20, validation_split=0.3, verbose=1, shuffle=False)
    return history, glucose_model


def train_test_split(glucose_data: pd.DataFrame):
    cleaned_data = {}
    for key in glucose_data.keys():
        cleaned_data[key] = glucose_data[key].__deepcopy__()
    cleaned_data = pd.DataFrame(cleaned_data)
    x = cleaned_data.loc[:, ["ts"]].values
    y = cleaned_data.loc[:, ["value"]].values
    x = StandardScaler().fit_transform(X=x)
    y = StandardScaler().fit_transform(y)
    cleaned_data["ts"] = x
    cleaned_data["value"] = y
    idx = round(len(glucose_data) * 0.8)
    train_x = cleaned_data[:idx]
    test_x = cleaned_data[idx:]
    return train_x, test_x


# ctrl+alt+shift+L REFORMATS CODE

if __name__ == "__main__":
    data, patient_data = load(TRAIN2_540_PATH)
    train, test = train_test_split(data["glucose_level"])
    history, model = model(train)
    print(history)
