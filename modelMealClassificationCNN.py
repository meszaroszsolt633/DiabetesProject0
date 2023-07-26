import pandas as pd
from defines import *
from functions import load_everything, drop_days_with_missing_glucose_data, drop_days_with_missing_eat_data, \
    fill_glucose_level_data_continuous, loadeveryxml
import numpy as np
from statistics import stdev
from scipy import signal
from model import create_dataset
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

def data_preparation(data: dict[str, pd.DataFrame], time_step: pd.Timedelta, missing_count_threshold,
                     missing_eat_threshold) -> dict[str, pd.DataFrame]:
    cleaned_data = drop_days_with_missing_glucose_data(data, missing_count_threshold)
    print('Glucose drop done')
    cleaned_data = drop_days_with_missing_eat_data(cleaned_data, missing_eat_threshold)
    print('Meal drop done')
    for key in cleaned_data.keys():
        cleaned_data[key] = cleaned_data[key].reset_index(drop=True)
    cleaned_data = fill_glucose_level_data_continuous(cleaned_data, time_step)
    print('Glucose fill done')
    return cleaned_data

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

def traintestsplitter(dataTrain: pd.DataFrame):
    dataTrain_70 = {}
    dataTrain_30 = {}

    # Define the split ratio (70% and 30%)
    split_ratio = 0.7

    # Loop through each key-value pair in the original dictionary
    for key, dataframe in dataTrain.items():
        # Calculate the number of rows to retain in the 70% data and 30% data
        total_rows = len(dataframe)
        rows_70 = int(total_rows * split_ratio)

        # Split the data based on the row indices
        data_70 = dataframe.iloc[:rows_70]
        data_30 = dataframe.iloc[rows_70:]

        # Store the split data into the new dictionaries
        dataTrain_70[key] = data_70
        dataTrain_30[key] = data_30
    return dataTrain_70,dataTrain_30

def model2(dataTrain, dataTest, lookback=50, maxfiltersize=10, epochnumber=50,modelnumber=1,learning_rate=0.001,oversampling=False):

    dataTrain,dataValidation=traintestsplitter(dataTrain)

    #TRAIN

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

    #TEST

    featurestest1 = dataTest['glucose_level']
    featurestest1.loc[:, 'carbs'] = ""
    featurestest1['carbs'] = featurestest1['carbs'].apply(lambda x: 0)

    featurestest2 = dataTest['meal']
    featurestest2 = featurestest2.drop(['type'], axis=1)
    featurestest2['carbs'] = featurestest2['carbs'].apply(lambda x: 1)

    featurestest = pd.concat([featurestest1, featurestest2])
    featurestest = featurestest.sort_values(by='ts', ignore_index=True)

    featurestesty = featurestest['carbs']
    featurestesty = ndimage.maximum_filter(featurestesty, size=maxfiltersize)
    featurestesty = pd.DataFrame(featurestesty)

    featurestestx = featurestest['value']
    featurestestx = pd.DataFrame(featurestestx)
    featurestestx = featurestestx.fillna(method='ffill')
    featurestestx['value'] = featurestestx['value'].astype('float64')


    #VALIDATION

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



    featuresvalidation  = pd.concat([features_validation_y, features_validation_x], axis=1)
    featurestrain=pd.concat([features_train_y,features_train_x],axis=1)
    featurestest=pd.concat([featurestesty,featurestestx],axis=1)

    featurestrain.columns = featurestrain.columns.astype(str)
    featuresvalidation.columns = featuresvalidation.columns.astype(str)
    featurestest.columns = featurestest.columns.astype(str)



    scaler = MinMaxScaler(feature_range=(0, 1))
    featurestrain=scaler.fit_transform(featurestrain)
    featuresvalidation=scaler.transform(featuresvalidation)
    featurestest=scaler.transform(featurestest)
    if(oversampling==True):
        trainY=featurestrain[:,0]
        trainX = featurestrain[:, 1]
        trainY = trainY.reshape(-1, 1)
        trainX = trainX.reshape(-1, 1)
        smote = SMOTE(random_state=42)
        trainX, trainY = smote.fit_resample(trainX, trainY)
        featurestrain = np.column_stack((trainY, trainX))
    trainX,trainY = create_dataset(featurestrain, lookback)
    validX,validY = create_dataset(featuresvalidation, lookback)
    testX,testY=create_dataset(featurestest,lookback)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    print("trainX:",trainX.shape)
    print("trainY:",trainY.shape)
    print("validX:",validX.shape)
    print("validY:", validY.shape)
    print("testX:", testX.shape)
    print("testY:", testY.shape)

    if(modelnumber==1):
        modelCNN(trainX, trainY, validX, validY,testX,testY, epochnumber,learning_rate)
    if(modelnumber==2):
        model_meal_RNN_1DCONV(trainX, trainY, validX, validY,testX,testY, epochnumber,learning_rate)


def modelCNN(train_x, train_y, validX, validY,testX,testY, epochnumber,learning_rate=0.001):

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

    test_loss, test_accuracy, test_precision, test_recall, test_auc, test_f1 = model.evaluate(testX, testY)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test AUC:", test_auc)
    print("Test F1 Score:", test_f1)

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

def model_meal_RNN_1DCONV(train_x, train_y, validX, validY,testX,testY, epochnumber,learning_rate=0.001):
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

    test_loss, test_accuracy, test_precision, test_recall, test_auc, test_f1 = model.evaluate(testX, testY)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test AUC:", test_auc)
    print("Test F1 Score:", test_f1)

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

def multiOutputModel(train_x, train_y, validX, validY,testX,testY, epochnumber,learning_rate=0.001):


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
    input = keras.Input(shape=(train_x.shape[1],train_x.shape[2]))
    conv1d1 = keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu')
    maxpool1 = keras.layers.MaxPooling1D(pool_size=2)
    dropout1 = keras.layers.Dropout(0.3)
    conv1d2 = keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')
    maxpool2 = keras.layers.MaxPooling1D(pool_size=2)
    dropout2 = keras.layers.Dropout(0.3)
    lstm1 = keras.layers.LSTM(128, return_sequences=True)
    dropout3 = keras.layers.Dropout(0.3)
    lstm2 = keras.layers.LSTM(64, return_sequences=False)
    dropout4 = keras.layers.Dropout(0.3)

    classLayer = keras.layers.Dense(1, activation='sigmoid')
    amountLayer = keras.layers.Dense(200, activation='softmax')

    x = conv1d1(input)
    x = maxpool1(x)
    x = dropout1(x)
    x = conv1d2(x)
    x = maxpool2(x)
    x = dropout2(x)
    x = lstm1(x)
    x = dropout3(x)
    x = lstm2(x)
    x = dropout4(x)
    output1 = classLayer(x)
    output2 = amountLayer(x)

    model = keras.Model(input = input, outputs = [output1, output2], name='MultiOutputLayer')

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy",
                                                                         tf.keras.metrics.Precision(name="precision"),
                                                                         tf.keras.metrics.Recall(name="recall"),
                                                                         tf.keras.metrics.AUC(name="auc"),
                                                                         tfa.metrics.F1Score(num_classes=1,
                                                                                             average='macro',
                                                                                             threshold=0.5)])

    history = model.fit(train_x, train_y, epochs=epochnumber, callbacks=[es_callback, reduce_lr, modelckpt_callback],
                        verbose=1, shuffle=False,
                        validation_data=(validX, validY))

    test_loss, test_accuracy, test_precision, test_recall, test_auc, test_f1 = model.evaluate(testX, testY)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test AUC:", test_auc)
    print("Test F1 Score:", test_f1)

    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train, patient_data = load(TRAIN2_544_PATH)
    test, patient_data = load(TEST2_544_PATH)
    #train, test= loadeveryxml()
    train = data_preparation(train, pd.Timedelta(5, "m"), 30, 3)
    test = data_preparation(test, pd.Timedelta(5, "m"), 30, 3)
    model2(dataTrain=train,dataTest=test,lookback=30,maxfiltersize=10,epochnumber=200,modelnumber=2,learning_rate=0.001,oversampling=False)



