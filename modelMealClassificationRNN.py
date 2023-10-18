import pandas as pd
from defines import *
from functions import load_everything, drop_days_with_missing_glucose_data, drop_days_with_missing_eat_data, \
    fill_glucose_level_data_continuous, loadeveryxml, create_dataset, create_dataset_multi, \
    create_variable_sliding_window_dataset, loadeverycleanedxml, write_model_stats_out_xml
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






#region MealDetectionModel
def model_mealdetection_RNN(dataTrain, dataValidation,  backward_slidingwindow, forward_slidingwindow, maxfiltersize=10, epochnumber=50, learning_rate=0.001, oversampling=False):
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
    model_meal_RNN(trainX, trainY, validX, validY, backward_slidingwindow, forward_slidingwindow, maxfiltersize, epochnumber, learning_rate, oversampling)


def model_meal_RNN(train_x, train_y, validX, validY, backward_slidingwindow, forward_slidingwindow, maxfiltersize, epochnumber, learning_rate, oversampling):
    model = keras.Sequential()
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
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

    write_model_stats_out_xml(history, validY, prediction, "RNN", backward_slidingwindow, forward_slidingwindow, maxfiltersize, learning_rate, oversampling)

#endregion

#region MultiOutputModel

def model_multi_RNN(dataTrain, dataValidation, lookback=50, maxfiltersize=10, epochnumber=50,  learning_rate=0.001, oversampling=False):
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

    features_train_y = pd.concat([features_train['eaten'], features_train['carbs']], axis=1)
    features_train_y = features_train_y.to_numpy()
    features_train_y = features_train_y.astype(float)
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

    features_validation_y = pd.concat([features_validation['eaten'], features_validation['carbs']], axis=1)
    features_validation_y = features_validation_y.to_numpy()
    features_validation_y = features_validation_y.astype(float)
    # Apply maximum filter separately to each column
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
    trainX, trainY = create_dataset_multi(featurestrain, lookback)
    validX, validY = create_dataset_multi(featuresvalidation, lookback)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))

    print("trainX:", trainX.shape)
    print("trainY:", trainY.shape)
    print("validX:", validX.shape)
    print("validY:", validY.shape)

    model_meal_RNN_multioutput(trainX, trainY, validX, validY,  epochnumber, learning_rate)
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
    #train_x = np.concatenate((train_x, train_x), axis=2)
    input1 = Input(shape=(train_x.shape[1], train_x.shape[2]))

    # Shared LSTM layers
    lstm1 = LSTM(512, return_sequences=True)(input1)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(256, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)
    lstm3 = LSTM(128, return_sequences=True)(dropout2)
    dropout3 = Dropout(0.3)(lstm3)
    lstm4 = LSTM(64, return_sequences=False)(dropout3)
    dropout4 = Dropout(0.3)(lstm4)

    # Output 1
    output1 = Dense(2, activation="sigmoid")(dropout4)

    # Output 2 (conditional output)

    # Create the model
    model = Model(inputs=[input1], outputs=[output1])
    model.summary()

    # Compile and train the model
    model.compile(loss="binary_crossentropy", optimizer=opt )

    history = model.fit([train_x], [train_y], epochs=epochnumber,
                                                                callbacks=[es_callback, reduce_lr, modelckpt_callback],
                                                                verbose=1, shuffle=False,
                                                                validation_data=([validX], [validY]))

    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3,0], label='prediction')
    plt.plot(validY[0:1440 * 3,0], label='test_data')
    plt.legend()
    plt.show()

#endregion

if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #train_1, test_1 = loadeverycleanedxml()
    train, patient_data = load(CLEANEDTRAIN_559_PATH)
    test, patient_data = load(CLEANEDTEST_559_PATH)

    #train, patient_data = load(TRAIN2_540_PATH)
    #test, patient_data = load(TEST2_540_PATH)

    #train = data_preparation(train, pd.Timedelta(5, "m"), 30, 3)
    #test = data_preparation(test, pd.Timedelta(5, "m"), 30, 3)

    model_mealdetection_RNN(dataTrain=train,dataValidation=test,backward_slidingwindow=3,forward_slidingwindow=15,maxfiltersize=10,epochnumber=5,learning_rate=0.001,oversampling=False)
