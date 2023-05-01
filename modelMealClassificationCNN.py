from keras.models import Sequential
from keras.layers import Conv1D,Conv2D, MaxPooling1D, Flatten, Dense, Dropout
import keras
import matplotlib.pyplot as plt

from model import create_dataset
from modelMealClassificationRNN import *

def model_base_CNN(dataTrain, dataValidation, lookback=50, maxfiltersize=10, epochnumber=50):


    #TRAIN

    feature_train1 = dataTrain['glucose_level']
    feature_train1['carbs'] = ""
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
    features_train_combined = pd.concat([features_train_y, features_train_x], axis=1)

    #VALIDATION

    feature_validation1 = dataValidation['glucose_level']
    feature_validation1['carbs'] = ""
    feature_validation1['carbs'] = feature_validation1['carbs'].apply(lambda x: 0)

    feature_validation2 = dataValidation['meal']
    feature_validation2 = feature_validation2.drop(['type'], axis=1)
    feature_validation2['carbs'] = feature_validation2['carbs'].apply(lambda x: 1)

    feature_validation = pd.concat([feature_validation1, feature_validation2])
    feature_validation = feature_validation.sort_values(by='ts', ignore_index=True)

    feature_validation_y = feature_validation['carbs']
    feature_validation_y = ndimage.maximum_filter(feature_validation_y, size=maxfiltersize)
    feature_validation_y = pd.DataFrame(feature_validation_y)

    feature_validation_x = feature_validation['value']
    feature_validation_x = pd.DataFrame(feature_validation_x)
    feature_validation_x = feature_validation_x.fillna(method='ffill')
    feature_validation_combined = pd.concat([feature_validation_y, feature_validation_x], axis=1)

    # Use if data separation is needed, use only one dataframe
    #train, valid = train_test_valid_split(features_train_combined)
    #trainX, trainY = create_dataset(train, look_back)
    #validX, validY = create_dataset(valid, look_back)

    scaler = MinMaxScaler(feature_range=(0, 1))

    features_train_combined = scaler.fit_transform(features_train_combined.values)
    feature_validation_combined = scaler.transform(feature_validation_combined.values)

    trainX, trainY = create_dataset(features_train_combined, lookback)
    validX, validY = create_dataset(feature_validation_combined, lookback)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))

    modelCNN(trainX, validX, validY, trainY,epochnumber)

def modelCNN(train_x, validX, validY, train_y,epochnumber):

    path_checkpoint = "modelMealCNN_checkpoint.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
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
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model.fit(train_x, train_y, epochs=epochnumber, callbacks=[ modelckpt_callback], verbose=1, shuffle=False,
              validation_data=(validX, validY))
    prediction = model.predict(validX)
    # Prediction and actual data plot
    # plt.figure(figsize=(20, 6))
    # plt.plot(prediction[0:1440 * 3], label='prediction')
    # plt.plot(validY[0:1440 * 3], label='test_data')
    # plt.legend()
    # plt.show()
    # Loss and validation loss plot
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training VS Validation loss')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    dataTrain, patient_data = load(TRAIN2_540_PATH)
    dataValidation, patient_data = load(TEST2_540_PATH)
    # clean_data = data_preparation(data, pd.Timedelta(5, "m"), 30, 3)
    model_base_CNN(dataTrain, dataValidation)