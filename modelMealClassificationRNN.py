import pandas as pd
from defines import *
from functions import load_everything, drop_days_with_missing_glucose_data, drop_days_with_missing_eat_data, \
    fill_glucose_level_data_continuous, loadeveryxml, create_dataset, create_dataset_multi, \
    create_variable_sliding_window_dataset, loadeverycleanedxml, write_model_stats_out_xml_classification, \
    loadeveryxmlparam, \
    dataPrepareRegression, data_cleaner, write_model_stats_out_xml_regression, expand_peak
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
from keras import backend as bcknd

def root_mean_squared_error(y_true, y_pred):
    return bcknd.sqrt(bcknd.mean(bcknd.square(y_pred - y_true)))




#region MealDetectionModel
def model_mealdetection_RNN(dataTrain, dataValidation, filename, backward_slidingwindow, forward_slidingwindow, maxfiltersize=10, epochnumber=50, learning_rate=0.001, oversampling=False):
    # TRAIN
    feature_train1 = dataTrain['glucose_level']
    feature_train1.loc[:, 'carbs'] = ""
    feature_train1['carbs'] = feature_train1['carbs'].apply(lambda x: 0)

    feature_train2 = dataTrain['meal']
    feature_train2 = feature_train2.drop(['type'], axis=1)
    feature_train2['carbs'] = feature_train2['carbs'].apply(lambda x: 1)

    features_train = pd.concat([feature_train1, feature_train2])
    features_train = features_train.sort_values(by='ts', ignore_index=True)

    features_train_y = features_train['carbs']
    features_train_y = ndimage.maximum_filter(features_train_y, size=maxfiltersize)
    features_train_y = pd.DataFrame(features_train_y)

    features_train_x = features_train['value']
    features_train_x = pd.DataFrame(features_train_x)
    features_train_x = features_train_x.fillna(method='ffill')
    features_train_x['value'] = features_train_x['value'].astype('float64')

    # VALIDATION

    feature_validation1 = dataValidation['glucose_level']
    feature_validation1.loc[:, 'carbs'] = ""
    feature_validation1['carbs'] = feature_validation1['carbs'].apply(lambda x: 0)

    feature_validation2 = dataValidation['meal']
    feature_validation2 = feature_validation2.drop(['type'], axis=1)
    feature_validation2['carbs'] = feature_validation2['carbs'].apply(lambda x: 1)

    features_validation = pd.concat([feature_validation1, feature_validation2])
    features_validation = features_validation.sort_values(by='ts', ignore_index=True)

    features_validation_y = features_validation['carbs']
    features_validation_y = ndimage.maximum_filter(features_validation_y, size=maxfiltersize)
    features_validation_y = pd.DataFrame(features_validation_y)

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
        trainY = featurestrain[:, 0]
        trainX = featurestrain[:, 1]
        trainY = trainY.reshape(-1, 1)
        trainX = trainX.reshape(-1, 1)
        smote = SMOTE(random_state=42)
        trainX, trainY = smote.fit_resample(trainX, trainY)
        featurestrain = np.column_stack((trainY, trainX))
    trainX, trainY = create_variable_sliding_window_dataset(featurestrain, backward_slidingwindow,
                                                            forward_slidingwindow)
    validX, validY = create_variable_sliding_window_dataset(featuresvalidation, backward_slidingwindow,
                                                            forward_slidingwindow)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))

    print("trainX:", trainX.shape)
    print("trainY:", trainY.shape)
    print("validX:", validX.shape)
    print("validY:", validY.shape)
    model_meal_RNN(trainX, trainY, validX, validY, filename, backward_slidingwindow, forward_slidingwindow, maxfiltersize, epochnumber, learning_rate, oversampling)


def model_meal_RNN(train_x, train_y, validX, validY, filename, backward_slidingwindow, forward_slidingwindow, maxfiltersize, epochnumber, learning_rate, oversampling):
    model = keras.Sequential()
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    path_checkpoint = "modelMealRNN_checkpoint.h5"
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=20)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    model.add(LSTM(256, return_sequences=True, activation="relu", input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
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

    write_model_stats_out_xml_classification(history, validY, prediction, filename, backward_slidingwindow, forward_slidingwindow, maxfiltersize, learning_rate, oversampling)

#endregion

#region MultiOutputModel

def model_regression_RNN(dataTrain, dataValidation, filename, backward_slidingwindow,forward_slidingwindow, epochnumber=50, scaling=True,learning_rate=0.001,oversampling=False,expansion_factor=4,expansion_multiplier=0.8):

    trainX, trainY, validX, validY = dataPrepareRegression(dataTrain, dataValidation, backward_slidingwindow, forward_slidingwindow, oversampling, expansion_factor, expansion_multiplier,scaling)


    model_meal_RNN_regression(trainX, trainY, validX, validY,  filename, epochnumber, backward_slidingwindow, forward_slidingwindow,  learning_rate, oversampling, scaling,expansion_factor,expansion_multiplier )
def model_meal_RNN_regression(train_x, train_y, validX, validY, filename, epochnumber,  backward_slidingwindow, forward_slidingwindow, learning_rate, oversampling, scaling, expansion_factor,expansion_multiplier):
    path_checkpoint = "modelMealRNN_checkpoint.h5"
    opt = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=30)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, activation="tanh", input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True, activation="tanh"))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True, activation="tanh"))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False, activation="tanh"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation=None))

    # Compile and train the model
    model.compile(loss="mse", optimizer=opt,metrics=["mae", "mse", root_mean_squared_error])

    history = model.fit(train_x, train_y, epochs=epochnumber,callbacks=[es_callback, reduce_lr, modelckpt_callback],
                                                                verbose=1, shuffle=False,
                                                                validation_data=([validX], [validY]))

    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()
    write_model_stats_out_xml_regression(history, validY, prediction, filename, backward_slidingwindow, forward_slidingwindow,  learning_rate, oversampling, scaling,expansion_factor,expansion_multiplier)
#endregion

if __name__ == "__main__":
    with tf.device("/cpu:0"):
     print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
     #train_1, test_1 = loadeverycleanedxml()
     #train, patient_data = load(CLEANEDTRAIN2_540_PATH)
     #test, patient_data = load(CLEANEDTEST2_540_PATH)

     train, test = loadeveryxmlparam(data_train3, data_test3)


     #train = data_cleaner(train, pd.Timedelta(5, "m"), 30, 3)
     #test = data_cleaner(test, pd.Timedelta(5, "m"), 30, 3)

     #model_mealdetection_RNN(dataTrain=train,dataValidation=test, filename="RNN_data3", backward_slidingwindow=2,forward_slidingwindow=20,maxfiltersize=16,epochnumber=200,learning_rate=0.001,oversampling=False)
     model_regression_RNN(dataTrain=train,\
                          dataValidation=test,\
                          filename="RNN_Regression_data3",\
                          backward_slidingwindow=2,\
                          forward_slidingwindow=25,\
                          epochnumber=100,\
                          learning_rate=0.001,\
                          oversampling=True,\
                          expansion_factor=5,\
                          expansion_multiplier=1,\
                          scaling=True)
