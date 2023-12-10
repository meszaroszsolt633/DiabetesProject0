import pandas as pd
from keras import backend as K

from defines import *
from functions import load_everything, drop_days_with_missing_glucose_data, drop_days_with_missing_eat_data, \
    fill_glucose_level_data_continuous, loadeveryxml, create_variable_sliding_window_dataset, data_cleaner, \
    loadeverycleanedxml, dataPrepareRegression, loadmultiplexml, write_model_stats_out_xml_classification, \
    write_all_cleaned_xml
import numpy as np
from statistics import stdev
from scipy import signal
from xml_read import load
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, roc_auc_score

from tensorflow import keras
import tensorflow as tf
from scipy import ndimage
from keras.models import Model
from keras import metrics, Sequential
from keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import tensorflow_addons as tfa
from keras.utils import plot_model


print("Is TensorFlow GPU accessible? ", tf.test.is_built_with_cuda())

# List the available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)



def dataPrepare(dataTrain, dataTest,patientID, backward_slidingwindow,forward_slidingwindow, maxfiltersize=10,oversampling=False):
    dataValidation = dataTest

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
    features_train_x = features_train_x.ffill()
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
    # Counting of 0 and 1 meal instances
    count_0 = (features_validation_y == 0).sum()
    count_1 = (features_validation_y == 1).sum()

    print(f"Count of 0: {count_0}")
    print(f"Count of 1: {count_1}")
    #
    features_validation_y = ndimage.maximum_filter(features_validation_y, size=maxfiltersize)
    features_validation_y = pd.DataFrame(features_validation_y)

    features_validation_x = features_validation['value']
    features_validation_x = pd.DataFrame(features_validation_x)
    features_validation_x = features_validation_x.ffill()
    features_validation_x['value'] = features_validation_x['value'].astype('float64')

    featuresvalidation = pd.concat([features_validation_y, features_validation_x], axis=1)
    featurestrain = pd.concat([features_train_y, features_train_x], axis=1)

    featurestrain.columns = featurestrain.columns.astype(str)
    #featurestrain['unique_id']=patientID
    featuresvalidation.columns = featuresvalidation.columns.astype(str)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(featurestrain[['value']])
    featurestrain['value'] = scaled_values
    featuresvalidation['value'] = scaler.transform(featuresvalidation[['value']])
    if (oversampling == True):
        featurestrain_values = featurestrain.values
        trainY = featurestrain_values[:, 0].reshape(-1, 1)
        trainX = featurestrain_values[:, 1].reshape(-1, 1)
        smote = SMOTE(random_state=42)
        trainX_resampled, trainY_resampled = smote.fit_resample(trainX, trainY.ravel())
        featurestrain = pd.DataFrame(np.column_stack((trainY_resampled, trainX_resampled)),
                                     columns=featurestrain.columns)

    print("Shape of trainX:", featurestrain.shape)
    print("Shape of trainY:", featuresvalidation.shape)
    mean_value = featurestrain['value'].mean()
    mean_validation_value= featuresvalidation['value'].mean()

    featurestrain['value'].fillna(mean_value, inplace=True)
    featuresvalidation['value'].fillna(mean_validation_value, inplace=True)

    trainX, trainY = create_variable_sliding_window_dataset(featurestrain, backward_slidingwindow,
                                                            forward_slidingwindow)
    validX, validY = create_variable_sliding_window_dataset(featuresvalidation, backward_slidingwindow,
                                                            forward_slidingwindow)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))


    print("Shape of trainX:", trainX.shape)
    print("Shape of trainY:", trainY.shape)
    print("Shape of validX:", validX.shape)
    print("Shape of validY:", validY.shape)

    return trainX, trainY, validX, validY


def model2(dataTrain, dataTest,patientID, backward_slidingwindow,forward_slidingwindow, maxfiltersize=10, epochnumber=50,learning_rate=0.001,oversampling=False):

    trainX, trainY, validX, validY = dataPrepare(dataTrain=dataTrain, dataTest=dataTest,patientID=patientID, backward_slidingwindow=backward_slidingwindow,forward_slidingwindow=forward_slidingwindow,maxfiltersize=maxfiltersize,oversampling=oversampling)

    print("trainX:",trainX.shape)
    print("trainY:",trainY.shape)
    print("validX:",validX.shape)
    print("validY:", validY.shape)
    modelCNN(train_x=trainX, train_y=trainY, validX=validX, validY=validY, epochnumber=epochnumber,learning_rate=learning_rate,backward_slidingwindow=backward_slidingwindow,forward_slidingwindow=forward_slidingwindow,maxfiltersize=maxfiltersize,oversampling=oversampling)


def modelCNN(train_x, train_y, validX, validY, epochnumber,learning_rate=0.001,backward_slidingwindow=3,forward_slidingwindow=15,maxfiltersize=15,oversampling=False):

    path_checkpoint = "modelMealCNN_checkpoint.h5"
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=30)

    class_weights = {0: 1.,
                     1: 3.}

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    model = Sequential()
    # Convolutional layers
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # LSTM layers
    model.add(LSTM(64, return_sequences=True,activation='relu'))
    model.add(Dropout(0.3))

    # Dense layers at the end
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",
                                                                         tf.keras.metrics.Precision(name="precision"),
                                                                         tf.keras.metrics.Recall(name="recall"),
                                                                         tf.keras.metrics.AUC(name="auc"),
                                                                         tfa.metrics.F1Score(num_classes=1,
                                                                                             average='weighted',
                                                                                             threshold=0.5)
                                                                         ])
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    history=model.fit(train_x, train_y, epochs=epochnumber,class_weight=class_weights, callbacks=[modelckpt_callback], verbose=1, shuffle=False,
              validation_data=(validX, validY))


    prediction = model.predict(validX)
    # Prediction and actual data plot
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()
    # Loss and validation loss plot
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training VS Validation loss')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    filename="R-CNN_Classification_ALL"
    #write_model_stats_out_xml_classification(history, validY, prediction, filename, backward_slidingwindow,forward_slidingwindow, maxfiltersize, learning_rate, oversampling)





if __name__ == "__main__":
     train, test = loadeveryxml()
    #train,test=loadmultiplexml(TRAIN2_FILE_PATHS,TEST2_FILE_PATHS)
    #train,_=load(TRAIN2_544_PATH)
    #test,_=load(TEST2_544_PATH)
     train = data_cleaner(train, pd.Timedelta(5, "m"), 25, 1)
     test = data_cleaner(test, pd.Timedelta(5, "m"), 25, 1)
     model2(dataTrain=train,dataTest=test,backward_slidingwindow=3,forward_slidingwindow=15,maxfiltersize=15,epochnumber=100,learning_rate=0.001,oversampling=False,patientID=1)



