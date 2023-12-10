import sklearn as sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import keras as keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Conv1D
#from modelMealClassificationTransformer import *
from keras.layers import MaxPooling1D
from keras.layers import Reshape
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, roc_auc_score
from functions import loadeverycleanedxml, create_variable_sliding_window_dataset, \
    write_model_stats_out_xml_classification, expand_peak, write_model_stats_out_xml_regression

print("Is TensorFlow GPU accessible? ", tf.test.is_built_with_cuda())

# List the available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
from keras import backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def msle(y_true, y_pred):
    return K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1)))
def dataPrepare_single(data, backward_slidingwindow,forward_slidingwindow, maxfiltersize=15):
    dataValidation = data

    feature_validation1 = dataValidation['glucose_level']
    feature_validation1.loc[:, 'carbs'] = ""
    feature_validation1['carbs'] = feature_validation1['carbs'].apply(lambda x: 0)

    feature_validation2 = dataValidation['meal']
    feature_validation2 = feature_validation2.drop(['type'], axis=1)
    #feature_validation2['carbs'] = feature_validation2['carbs'].apply(lambda x: 1) #Classification
    feature_validation2['carbs'] = feature_validation2['carbs'].apply(lambda x: x) #Regression

    features_validation = pd.concat([feature_validation1, feature_validation2])
    features_validation = features_validation.sort_values(by='ts', ignore_index=True)

    features_validation_y = features_validation['carbs']
    #features_validation_y = ndimage.maximum_filter(features_validation_y, size=maxfiltersize) #Classification
    features_validation_y = pd.DataFrame(features_validation_y)
    features_validation_y= features_validation_y.values.astype(float)
    features_validation_y = expand_peak(features_validation_y, 6, 1.0)
    features_validation_y = pd.DataFrame(features_validation_y)

    features_validation_x = features_validation['value']
    features_validation_x = pd.DataFrame(features_validation_x)
    features_validation_x = features_validation_x.ffill()
    features_validation_x['value'] = features_validation_x['value'].astype('float64')

    featuresvalidation = pd.concat([features_validation_y, features_validation_x], axis=1)
    featuresvalidation.columns = featuresvalidation.columns.astype(str)
    validX, validY = create_variable_sliding_window_dataset(featuresvalidation, backward_slidingwindow,
                                                            forward_slidingwindow)

    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))
    return validX, validY

def dataPrepare_single_regression(dataTrain, backward_slidingwindow, forward_slidingwindow,expansion_factor,expansion_multiplier,scaling):
    # TRAIN
    dataValidation = dataTrain

    feature_validation1 = dataValidation['glucose_level']
    feature_validation1.loc[:, 'carbs'] = ""
    feature_validation1['carbs'] = feature_validation1['carbs'].apply(lambda x: 0)

    feature_validation2 = dataValidation['meal']
    feature_validation2 = feature_validation2.drop(['type'], axis=1)
    # feature_validation2['carbs'] = feature_validation2['carbs'].apply(lambda x: 1) #Classification
    feature_validation2['carbs'] = feature_validation2['carbs'].apply(lambda x: x)  # Regression

    features_validation = pd.concat([feature_validation1, feature_validation2])
    features_validation = features_validation.sort_values(by='ts', ignore_index=True)

    features_validation_y = features_validation['carbs']
    # features_validation_y = ndimage.maximum_filter(features_validation_y, size=maxfiltersize) #Classification
    features_validation_y = pd.DataFrame(features_validation_y)
    features_validation_y = features_validation_y.values.astype(float)
    features_validation_y = expand_peak(features_validation_y, 6, 1.0)
    features_validation_y = pd.DataFrame(features_validation_y)

    features_validation_x = features_validation['value']
    features_validation_x = pd.DataFrame(features_validation_x)
    features_validation_x = features_validation_x.ffill()
    features_validation_x['value'] = features_validation_x['value'].astype('float64')

    featuresvalidation = pd.concat([features_validation_y, features_validation_x], axis=1)
    featuresvalidation.columns = featuresvalidation.columns.astype(str)
    validX, validY = create_variable_sliding_window_dataset(featuresvalidation, backward_slidingwindow,
                                                            forward_slidingwindow)

    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))
    return validX, validY


alltrain, alltest=loadeverycleanedxml()
#X_test,y_test=dataPrepare_single(data=alltest, backward_slidingwindow=3, forward_slidingwindow=15,maxfiltersize=15)
X_test,y_test=dataPrepare_single_regression(dataTrain=alltest, backward_slidingwindow=3, forward_slidingwindow=15,expansion_factor=7,expansion_multiplier=1.0,scaling=False)
cols = ['Name','Time','Iteration','CH','BloodSugar']
data = pd.read_parquet('converted.parquet', columns=cols)
#data=data.head(10000)




scale = MinMaxScaler(feature_range = (0,1))

label_encoder = LabelEncoder()
data.loc[:, "Name"] = label_encoder.fit_transform(data.loc[:, "Name"])

#for train data
dataX = data[['Name','Time','Iteration','CH','BloodSugar']]
dataX = dataX[dataX.Iteration < 9]
#dataX = dataX[dataX.Name < 1]
dataY = dataX['CH']
dataY = dataY.apply(lambda x: x if x > 1 else 0)
dataXX = dataX.drop(['Name','Time','Iteration','CH'],axis=1)


count_0 = (dataY == 0).sum()
count_1 = (dataY > 0).sum()

print(f"Count of 0: {count_0}")
print(f"Count of 1: {count_1}")

from scipy import ndimage
dataY = ndimage.maximum_filter(dataY,size=15)

# from scipy import signal
# window = signal.windows.gaussian(15,1)
# dataY = np.convolve(dataY,window)


X_train = []
y_train = []

time_step = 15
time_step_forward = 75

names = dataX.Name.unique()
iterations = dataX.Iteration.unique()

for k in range(0,len(iterations)*len(names)):
    for i in range(time_step,1440-time_step_forward):
        X_train.append(dataXX[(i+(k*1440))-time_step:(i+(k*1440))+time_step_forward])
        y_train.append(dataY[(i+(k*1440)):(i+(k*1440))+1])
X_train = np.array(X_train)
y_train = np.array(y_train)

count_0 = (y_train == 0).sum()
count_1 = (y_train == 1).sum()

print(f"Count of 0: {count_0}")
print(f"Count of 1: {count_1}")

X_train=X_train[:,::5,:]

##for test data
#testdataX = data[['Name','Time','Iteration','CH','BloodSugar']]
#testdataX = testdataX[testdataX.Iteration > 8]
##testdataX = testdataX[testdataX.Name < 1]
#testdataY = testdataX['CH']
#testdataY = testdataY.apply(lambda x: 1 if x > 1 else 0)
#testdataXX = testdataX.drop(['Name','Time','Iteration','CH'],axis=1)
#
#
#testdataY = ndimage.maximum_filter(testdataY,size=15)
#
## window = signal.windows.gaussian(15,1)
## testdataY = np.convolve(testdataY,window)
#
#namestest = testdataX.Name.unique()
#iterationstest = testdataX.Iteration.unique()
#
#X_test = []
#y_test = []
#
#for z in range(0,len(iterationstest)*len(namestest)):
#    for i in range(time_step, 1440-time_step_forward):
#        X_test.append(testdataXX[(i+(z*1440))-time_step:(i+(z*1440))+time_step_forward])
#        y_test.append(testdataY[(i+(z*1440)):(i+(z*1440))+1])
#
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# nsamples, nx, ny = X_train.shape
# X_train = X_train.reshape(nsamples,(nx*ny))

# nsamples, nx, ny = X_test.shape
# X_test = X_test.reshape(nsamples,(nx*ny))

# X_train = scale.fit_transform(X_train)

# X_test = scale.transform(X_test)

y_train = np.reshape(y_train, (-1,1))
# y_train = scale.fit_transform(y_train)

y_test = np.reshape(y_test, (-1,1))


x_train = np.array(X_train)
x_test = np.array(X_test)
print(type(X_train))
print(X_train.shape)
has_nan = np.isnan(X_train).any()

print(has_nan)
input_shape = x_train.shape[1:]


#model = Sequential()
##model.add(Conv1D(filters=160,kernel_size=7,kernel_initializer="truncated_normal",input_shape=(X_train.shape[1],1)))
## model.add(LSTM(192,return_sequences=False,activation="tanh",input_shape=(X_train.shape[1],1)))
## #model.add(Dropout(0.25))
## #model.add(Dense(128))
## model.add(Dense(1,activation="relu"))
##class_weights = {0: 1.,
##                     1: 3.}
#opt = keras.optimizers.Adam(learning_rate=0.001)
#
#model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dropout(0.3))
#model.add(LSTM(128, return_sequences=True))
#model.add(Dropout(0.3))
#model.add(LSTM(64, return_sequences=False))
#model.add(Dropout(0.3))
#model.add(Dense(1, activation="linear"))
#model.summary()
#model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae", "mse",rmse,msle])
#
#history = model.fit( x=X_train, y=y_train, epochs=100,validation_data=(X_test, y_test))
#
#import matplotlib.pyplot as plt
#
#prediction = model.predict(X_test)
#
##prediction = scale.inverse_transform(prediction)
#
#plt.figure(figsize=(20,6))
#plt.plot(prediction[0:1440*3], label='prediction')
#plt.plot(y_test[0:1440*3], label='test_data')
#plt.legend()
#plt.show()
#
## Loss and validation loss plot
#plt.plot(history.history['loss'], label='Training loss')
#plt.plot(history.history['val_loss'], label='Validation loss')
#plt.title('Training VS Validation loss')
#plt.xlabel('No. of Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res
n_classes=1
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)  # Linear activation for regression
    return keras.Model(inputs, outputs)

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",  # Change to regression loss
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mae", "mse"]  # Add appropriate regression metrics
)

model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)]

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=2000,
    batch_size=64,
    callbacks=callbacks,
)

prediction = model.predict(x_test)

# Plotting
plt.figure(figsize=(20, 6))
plt.plot(prediction[0:1440 * 3], label='prediction')
plt.plot(y_test[0:1440 * 3], label='test_data')
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

model.evaluate(x_test, y_test, verbose=1)


filename='modelproject_01_Transformer_regression'
write_model_stats_out_xml_regression(history, y_test, prediction, filename, 3, 15,  0.001, False, False,7,1)


