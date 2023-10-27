import pandas as pd
import numpy as np
cols = ['Name','Time','Iteration','CH','BloodSugar']
data = pd.read_csv('t1dpatient_ALLpatient_1min.csv',usecols=cols)


import sklearn as sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range = (0,1))

label_encoder = LabelEncoder()
data.loc[:, "Name"] = label_encoder.fit_transform(data.loc[:, "Name"])

#for train data
dataX = data[['Name','Time','Iteration','CH','BloodSugar']]
dataX = dataX[dataX.Iteration < 9]
#dataX = dataX[dataX.Name < 1]
dataY = dataX['CH']
dataY = dataY.apply(lambda x: 1 if x > 1 else 0)
dataXX = dataX.drop(['Name','Time','Iteration','CH'],axis=1)

from scipy import ndimage
dataY = ndimage.maximum_filter(dataY,size=15)

# from scipy import signal
# window = signal.windows.gaussian(15,1)
# dataY = np.convolve(dataY,window)


X_train = []
y_train = []

time_step = 15
time_step_forward = 50

names = dataX.Name.unique()
iterations = dataX.Iteration.unique()

for k in range(0,len(iterations)*len(names)):
    for i in range(time_step,1440-time_step_forward):
        X_train.append(dataXX[(i+(k*1440))-time_step:(i+(k*1440))+time_step_forward])
        y_train.append(dataY[(i+(k*1440)):(i+(k*1440))+1])

#for test data
testdataX = data[['Name','Time','Iteration','CH','BloodSugar']]
testdataX = testdataX[testdataX.Iteration > 8]
#testdataX = testdataX[testdataX.Name < 1]
testdataY = testdataX['CH']
testdataY = testdataY.apply(lambda x: 1 if x > 1 else 0)
testdataXX = testdataX.drop(['Name','Time','Iteration','CH'],axis=1)


testdataY = ndimage.maximum_filter(testdataY,size=15)

# window = signal.windows.gaussian(15,1)
# testdataY = np.convolve(testdataY,window)

namestest = testdataX.Name.unique()
iterationstest = testdataX.Iteration.unique()

X_test = []
y_test = []

for z in range(0,len(iterationstest)*len(namestest)):
    for i in range(time_step, 1440-time_step_forward):
        X_test.append(testdataXX[(i+(z*1440))-time_step:(i+(z*1440))+time_step_forward])
        y_test.append(testdataY[(i+(z*1440)):(i+(z*1440))+1])

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

import keras as keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Reshape
from keras.callbacks import EarlyStopping

model = Sequential()

#model.add(Conv1D(filters=160,kernel_size=7,kernel_initializer="truncated_normal",input_shape=(X_train.shape[1],1)))
# model.add(LSTM(192,return_sequences=False,activation="tanh",input_shape=(X_train.shape[1],1)))
# #model.add(Dropout(0.25))
# #model.add(Dense(128))
# model.add(Dense(1,activation="relu"))

opt = keras.optimizers.Adam(learning_rate=0.01)

model.add(LSTM(128,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128,return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy",keras.metrics.Precision(), keras.metrics.Recall()])
history = model.fit(X_train,y_train, epochs=25)


import matplotlib.pyplot as plt

prediction = model.predict(X_test)

#prediction = scale.inverse_transform(prediction)

plt.figure(figsize=(20,6))
plt.plot(prediction[0:1440*3], label='prediction')
plt.plot(y_test[0:1440*3], label='test_data')
plt.legend()
plt.show()

result = model.evaluate(X_test,y_test)
print(result)

print('loss: ' + str(result[0]) + ', Accuracy: ' + str(result[1]))


from tensorflow.keras.models import save_model
save_model(model, "model.h5")

import matplotlib.pyplot as plt
plt.figure(figsize=(20,6))
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()