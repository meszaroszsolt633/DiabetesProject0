from keras.models import Sequential
from keras.layers import Conv1D,Conv2D, MaxPooling1D, Flatten, Dense, Dropout
import keras
import matplotlib.pyplot as plt
from modelMealClassification import *

def modelbaseCNN(data, windowsize):
    feature1=data['glucose_level']
    feature1['carbs']=""
    feature1['carbs']=feature1['carbs'].apply(lambda x:0)
    feature2=data['meal']
    feature2=feature2.drop(['type'],axis=1)

    feature2['carbs'] = feature2['carbs'].apply(lambda x: 1)
    features = pd.concat([feature1, feature2])
    features = features.sort_values(by='ts', ignore_index=True)
    featureY=features['carbs']
    featureX=features['value']
    featureY = ndimage.maximum_filter(featureY, size=10)
    featureY=pd.DataFrame(featureY)
    featureX=pd.DataFrame(featureX)
    featureX=featureX.fillna(method='ffill')
    features2=pd.concat([featureY,featureX],axis=1)

    look_back=windowsize
    train, valid = train_test_valid_split(features2)
    trainX, trainY = create_dataset(train, look_back)
    validX, validY = create_dataset(valid, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0],  trainX.shape[1],1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1],1))

    modelCNN(trainX, validX, validY, trainY, look_back)

def modelCNN(train_x, validX, validY, train_y, look_back):

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
    # Add a 1D convolution layer with 64 filters and kernel size of 3
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(train_x.shape[1],train_x.shape[2])))
    # Add a max pooling layer with pool size of 2
    model.add(MaxPooling1D(pool_size=2))
    # Add a dropout layer with dropout rate of 0.3
    model.add(Dropout(0.3))
    # Add another 1D convolution layer with 32 filters and kernel size of 3
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    # Add a max pooling layer with pool size of 2
    model.add(MaxPooling1D(pool_size=2))
    # Add a dropout layer with dropout rate of 0.3
    model.add(Dropout(0.3))
    # Flatten the output of the previous layer
    model.add(Flatten())
    # Add a dense layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))
    # Add a dropout layer with dropout rate of 0.3
    model.add(Dropout(0.3))
    # Add a final dense layer with a single output unit and sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model with binary cross-entropy loss and Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model on your data
    model.fit(train_x, train_y, epochs=1000, callbacks=[es_callback, modelckpt_callback], verbose=1, shuffle=False,
              validation_data=(validX, validY))
    # Make predictions on the validation set
    prediction = model.predict(validX)
    # Plot the predictions against the actual values
    plt.figure(figsize=(20, 6))
    plt.plot(prediction[0:1440 * 3], label='prediction')
    plt.plot(validY[0:1440 * 3], label='test_data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data, patient_data = load(TRAIN2_540_PATH)
    #clean_data = data_preparation(data, pd.Timedelta(5, "m"), 30, 3)
    modelbaseCNN(data,60)