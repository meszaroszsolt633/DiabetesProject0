import pandas as pd
from defines import *
from model import *
from statistics import stdev
from scipy import signal
from xml_read import *
from xml_write import *
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
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


def model(data: dict[str,pd.DataFrame]):
    scaler=MinMaxScaler(feature_range=(0,1))
    split_fraction = 0.8
    train_split = int(split_fraction * int(data['glucose_level'].shape[0]))
    step = 3
    past = 15
    future = 50
    learning_rate = 0.001
    batch_size = 128
    epochs = 50
    dataX = data['glucose_level']
    dataX['carbs'] = ""
    dataX['carbs'] = dataX['carbs'].apply(lambda x: 0)


    dataY= dataX.drop(['value'], axis=1)
    mealdata=data['meal']
    mealdata=mealdata.drop(['type'],axis=1)
    mealdata['carbs']=mealdata['carbs'].apply(lambda x: 1)
    meal = pd.concat([dataY, mealdata])
    meal = meal.sort_values(by='ts', ignore_index=True)
    dataY=meal
    dataX = dataX.drop(['carbs'], axis=1)

    dataX['ts'] = pd.to_numeric(pd.to_datetime(dataX['ts']))
    dataY['ts'] = pd.to_numeric(pd.to_datetime(dataY['ts']))
    dataX = scaler.fit_transform(dataX)
    dataY = scaler.transform(dataY)
    dataY=pd.DataFrame(data=dataY)
    dataX=pd.DataFrame(data=dataX)


    train_data=dataX.loc[0:train_split-1]
    val_data=dataX.loc[train_split:]

    #Training dataset
    start=past+future
    end=start+train_split
    rng1=train_data.shape[1]
    x_train = train_data[[i for i in range(rng1)]].values
    y_train = dataY.iloc[start:end][[1]]
    sequence_length = int(past / step)
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
    #Validation dataset
    x_end = len(val_data) - past - future
    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end][[i for i in range(rng1)]].values
    y_val = dataY.iloc[label_start:][[1]]

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch
    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()

    #TRAIN

    path_checkpoint = "model_checkpoint.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )

    visualize_loss(history, "Training and Validation Loss")

    for x, y in dataset_val.take(5):
        show_plot(
            [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
            12,
            "Single Step Prediction",
        )
def model2(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    split_fraction=0.8
    train_split = int(split_fraction * int(data['glucose_level'].shape[0]))
    glucoselevel = data['glucose_level']
    meal = data['meal']
    dataXX=glucoselevel
    dataXX['carbs']=""
    dataXX['carbs']=dataXX['carbs'].apply(lambda x: 0)
    dataY = pd.concat([dataXX,meal])
    dataY = dataY.drop(['type'], axis=1)
    dataY = dataY.drop(['value'], axis=1)
    dataY=dataY.sort_values(by=['ts'],ignore_index=True)

    dataXX=dataXX.drop(['ts'],axis=1)
    dataXX=dataXX.drop(['carbs'],axis=1)
    dataY=dataY.drop(['ts'],axis=1)
    dataY=dataY.iloc[:dataY.shape[0]-51,:]
    #dataXX['ts'] = pd.to_numeric(pd.to_datetime(dataXX['ts']))
    #dataY['ts'] = pd.to_numeric(pd.to_datetime(dataY['ts']))

    dataY = pd.DataFrame(data=dataY)
    dataXX = pd.DataFrame(data=dataXX)

    train_data = dataXX.loc[0:train_split - 1]
    val_data = dataXX.loc[train_split:]
    past = 15
    future = 50
    start = past + future
    end = start + train_split
    rng1 = train_data.shape[1]
    X_train = train_data.values
    y_train = dataY.iloc[start:end]

    x_end = len(val_data) - past - future
    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end].values
    y_val = dataY.iloc[label_start:].values

    X_train[:] = X_train[:].astype(float)
    x_val[:] = x_val[:].astype(float)

    y_train['carbs'] = y_train['carbs'].astype(float)
    y_val[:]=y_val[:].astype(float)

    y_train['carbs']=y_train['carbs'].apply(lambda x:1 if x>0 else 0)

    window = signal.windows.gaussian(15, 1)

    y_train=np.ravel(y_train)
    y_val=np.ravel(y_val)

    y_train = np.convolve(y_train, window)
    y_val = np.convolve(y_val, window)

    y_train = y_train[:y_train.shape[0] - 14]
    y_val = y_val[:y_val.shape[0] - 14]

    X_train = np.reshape(X_train, (-1, 1))
    x_val = np.reshape(x_val, (-1, 1))
    y_train = np.reshape(y_train, (-1, 1))
    y_val = np.reshape(y_val, (-1, 1))
    X_train = scaler.fit_transform(X_train)
    y_train = scaler.transform(y_train)
    x_val=scaler.transform(x_val)
    y_val=scaler.transform(y_val)
    #y_train=np.reshape(y_train,(y_train.shape[0],1))


    model = keras.models.Sequential()
    model.add(
        keras.layers.Conv1D(filters=160, kernel_size=1, kernel_initializer="truncated_normal", input_shape=(1,X_train.shape[1])))
    model.add(LSTM(192, return_sequences=False, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    # model.add(Dense(128))
    model.add(Dense(1, activation="softmax"))
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50)










if __name__ == "__main__":
    data, patient_data = load(TRAIN2_540_PATH)
    clean_data = data_preparation(data, pd.Timedelta(5, "m"), 30, 3)
    model2(clean_data)