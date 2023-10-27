from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersDF
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, RNN
import pandas as pd
from xml_read import load
from defines import *
from scipy import ndimage
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn import datasets

import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NHITS, LSTM
from neuralforecast.losses.pytorch import DistributionLoss, Accuracy

# Defined in neuralforecast.utils

dataTrain, patient_data = load(TRAIN2_544_PATH)

maxfiltersize = 10

feature_train1 = dataTrain['glucose_level']
feature_train1.loc[:, 'carbs'] = ""
feature_train1['carbs'] = feature_train1['carbs'].apply(lambda x: 0)

feature_train2 = dataTrain['meal']
feature_train2 = feature_train2.drop(['type'], axis=1)
feature_train2['carbs'] = feature_train2['carbs'].apply(lambda x: 1)
features_train = pd.concat([feature_train1, feature_train2])
features_train = features_train.sort_values(by='ts', ignore_index=True)
features_train_y = features_train['carbs']
features_train['carbs'] = ndimage.maximum_filter(features_train['carbs'], size=maxfiltersize)
features_train['carbs'] = features_train['carbs'].astype('float64')
features_train['value'] = features_train['value'].astype('float64')
features_train['value'] = features_train['value'].fillna((features_train['value'].shift() + features_train['value'].shift(-1)) / 2)
features_train['value'].fillna(features_train['value'].mean(), inplace=True)
features_train['carbs'].fillna(features_train['carbs'].mean(), inplace=True)

Y_df=features_train
Y_df = Y_df.rename(columns={
    'carbs': 'y',
    'ts': 'ds',
    'value': 'unique_id'
})
print("hello")


horizon = 12

# Try different hyperparmeters to improve accuracy.
models = [MLP(h=horizon,                           # Forecast horizon
              input_size=2 * horizon,              # Length of input sequence
              loss=DistributionLoss('Bernoulli'),  # Binary classification loss
              valid_loss=Accuracy(),               # Accuracy validation signal
              max_steps=500,                       # Number of steps to train
              scaler_type='standard',              # Type of scaler to normalize data
              hidden_size=64,                      # Defines the size of the hidden state of the LSTM
              #early_stop_patience_steps=2,         # Early stopping regularization patience
              val_check_steps=10,                  # Frequency of validation signal (affects early stopping)
              ),
          NHITS(h=horizon,                          # Forecast horizon
                input_size=2 * horizon,             # Length of input sequence
                loss=DistributionLoss('Bernoulli'), # Binary classification loss
                valid_loss=Accuracy(),              # Accuracy validation signal
                max_steps=500,                      # Number of steps to train
                n_freq_downsample=[2, 1, 1],        # Downsampling factors for each stack output
                #early_stop_patience_steps=2,        # Early stopping regularization patience
                val_check_steps=10,                 # Frequency of validation signal (affects early stopping)
                )
          ]
nf = NeuralForecast(models=models, freq='M')
nf.fit(df=Y_df)

Y_hat_df = nf.predict()

Y_hat_df = Y_hat_df.reset_index()
Y_hat_df.head()


fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = pd.concat([Y_df, Y_hat_df]).set_index('ds') # Concatenate the train and forecast dataframes
plot_df[['y', 'MLP', 'NHITS']].plot(ax=ax, linewidth=2)

ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
plt.show()