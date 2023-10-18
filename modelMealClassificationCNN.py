import pandas as pd
from defines import *
from functions import load_everything, drop_days_with_missing_glucose_data, drop_days_with_missing_eat_data, \
    fill_glucose_level_data_continuous, loadeveryxml, create_variable_sliding_window_dataset, data_preparation, \
    loadeverycleanedxml
import numpy as np
from statistics import stdev
from scipy import signal
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
from functions import dataPrepare






def model2(dataTrain, dataTest, backward_slidingwindow,forward_slidingwindow, maxfiltersize=10, epochnumber=50,modelnumber=1,learning_rate=0.001,oversampling=False):

    trainX, trainY, validX, validY = dataPrepare(dataTrain, dataTest, backward_slidingwindow,forward_slidingwindow)

    print("trainX:",trainX.shape)
    print("trainY:",trainY.shape)
    print("validX:",validX.shape)
    print("validY:", validY.shape)

    if(modelnumber==1):
        modelCNN(trainX, trainY, validX, validY, epochnumber,learning_rate)
    if(modelnumber==2):
        model_meal_RNN_1DCONV(trainX, trainY, validX, validY, epochnumber,learning_rate)


def modelCNN(train_x, train_y, validX, validY, epochnumber,learning_rate=0.001):

    path_checkpoint = "modelMealCNN_checkpoint.h5"
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=30)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    class_weights = {0: 1.,
                     1: 2.}

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(train_x.shape[1],train_x.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",
                                                                         tf.keras.metrics.Precision(name="precision"),
                                                                         tf.keras.metrics.Recall(name="recall"),
                                                                         tf.keras.metrics.AUC(name="auc"),
                                                                         tfa.metrics.F1Score(num_classes=1,
                                                                                             average='macro',
                                                                                             threshold=0.5)
                                                                         ])
    history=model.fit(train_x, train_y, epochs=epochnumber,class_weight=class_weights, callbacks=[ es_callback,reduce_lr,modelckpt_callback], verbose=1, shuffle=False,
              validation_data=(validX, validY))


    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()
    # Loss and validation loss plot
   #plt.plot(history.history['loss'], label='Training loss')
   #plt.plot(history.history['val_loss'], label='Validation loss')
   #plt.title('Training VS Validation loss')
   #plt.xlabel('No. of Epochs')
   #plt.ylabel('Loss')
   #plt.legend()
   #plt.show()

def model_meal_RNN_1DCONV(train_x, train_y, validX, validY, epochnumber,learning_rate=0.001):
    model = keras.Sequential()

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    path_checkpoint = "modelMeal_checkpoint.h5"
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=40)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    # Conv1D layers
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(train_x.shape[1],train_x.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # LSTM layers
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))

    # Dense output layer
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",
                                                                         tf.keras.metrics.Precision(name="precision"),
                                                                         tf.keras.metrics.Recall(name="recall"),
                                                                         tf.keras.metrics.AUC(name="auc"),
                                                                         tfa.metrics.F1Score(num_classes=1,
                                                                                             average='macro',
                                                                                             threshold=0.5)
                                                                         ])
    history = model.fit(train_x, train_y, epochs=epochnumber, callbacks=[es_callback,reduce_lr,modelckpt_callback], verbose=1, shuffle=False,
              validation_data=(validX, validY))


    prediction = model.predict(validX)
    #Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()
    #Loss and validation loss plot
   #plt.plot(history.history['loss'],  label='Training loss')
   #plt.plot(history.history['val_loss'],  label='Validation loss')
   #plt.title('Training VS Validation loss')
   #plt.xlabel('No. of Epochs')
   #plt.ylabel('Loss')
   #plt.legend()
   #plt.show()




if __name__ == "__main__":
   #train, patient_data = load(TRAIN2_544_PATH)
   #test, patient_data = load(TEST2_544_PATH)
    train,test=loadeverycleanedxml()
   #train = data_preparation(train, pd.Timedelta(5, "m"), 30, 3)

   #test = data_preparation(test, pd.Timedelta(5, "m"), 30, 3)

    model2(dataTrain=train,dataTest=test,backward_slidingwindow=3,forward_slidingwindow=15,maxfiltersize=15,epochnumber=100,modelnumber=1,learning_rate=0.001,oversampling=False)



