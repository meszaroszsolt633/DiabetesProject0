from neuralforecast.core import NeuralForecast
from neuralforecast.models import Informer, Autoformer, FEDformer, PatchTST
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from neuralforecast.losses.numpy import mae
from neuralforecast.losses.pytorch import DistributionLoss, Accuracy
import torch
import torch.nn as nn

from xml_read import load
from functions import loadmultiplexml, create_variable_sliding_window_dataset_multiple_features
from modelMealClassificationCNN import dataPrepare
from defines import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


import matplotlib.pyplot as plt
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
#train, test = loadmultiplexml(trainfilepaths=CLEANEDTRAIN_FILE_PATHS, testfilepaths=CLEANEDTEST_FILE_PATHS)
#train_x, train_y, validX, validY = dataPrepare(train, test, 3, 15, 15, False)
i = [0, 1, 3]
targeti = 2

def expand_peak(arr, expansion_factor=2, expansion_multiplier=0.8):
    expanded_arr = [0] * len(arr)

    for i in range(len(arr)):
        if arr[i] != 0:
            expanded_peak = [0] * (expansion_factor * 2 + 1)
            expanded_peak[expansion_factor] = arr[i]

            for j in range(1, expansion_factor + 1):
                # Calculate the multiplier for each step
                multiplier = expansion_multiplier ** j
                # Check if the indices are within bounds before assignment
                if 0 <= i - j < len(expanded_arr):
                    expanded_arr[i - j] = arr[i] * multiplier
                if 0 <= i + j < len(expanded_arr):
                    expanded_arr[i + j] = arr[i] * multiplier

            expanded_arr[i] = arr[i]

    return expanded_arr
def dataset_creator(dataTrain,maxfiltersize,patientid):
    feature_train1 = dataTrain['glucose_level']
    feature_train1.loc[:, 'carbs'] = ""
    feature_train1['carbs'] = feature_train1['carbs'].apply(lambda x: 0)

    feature_train2 = dataTrain['meal']
    feature_train2 = feature_train2.drop(['type'], axis=1)
    feature_train2['carbs'] = feature_train2['carbs'].apply(lambda x: x)

    features_train = pd.concat([feature_train1, feature_train2])
    features_train = features_train.sort_values(by='ts', ignore_index=True)

    features_train_y = features_train['carbs']
    features_train_y =features_train_y.astype(int)
    features_train_y = expand_peak(features_train_y,5, 1)
    features_train_y = pd.DataFrame(features_train_y)

    features_train_ds=features_train['ts']

    features_train_x = features_train['value']
    features_train_x = pd.DataFrame(features_train_x)
    features_train_x = features_train_x.ffill()
    features_train_x['value'] = features_train_x['value'].astype('float64')
    featurestrain = pd.concat([features_train_y, features_train_x,features_train_ds], axis=1)
    featurestrain['unique_id'] = 559
    featurestrain['unique_id'] = featurestrain['unique_id'] .astype(int)
    featurestrain = featurestrain.rename(columns={0: 'y', 'ts': 'ds'})

    total_rows = len(featurestrain)
    train_idx = int(total_rows * 0.6)
    val_idx = train_idx + int(total_rows * 0.2)

    # Splitting the dataset sequentially
    train = featurestrain.iloc[:train_idx]
    validation = featurestrain.iloc[train_idx:val_idx]
    test = featurestrain.iloc[val_idx:]

    #scaler = MinMaxScaler(feature_range=(0, 1))
   #scaled_values = scaler.fit_transform(featurestrain[['value']])
   #featurestrain['value'] = scaled_values
    return train, validation, test

Y_df_train=[]
Y_df_test=[]
Y_df_validation=[]
Y_df=[]
for patient in CLEANEDALL_FILE_PATHS:
    dataset_a, patientdata = load(patient)
    patient_df_train,patient_df_validation,patient_df_test= dataset_creator(dataset_a, 15, patientdata['id'])
    Y_df_train.append(patient_df_train)
    Y_df_test.append(patient_df_test)
    Y_df_validation.append(patient_df_validation)
Y_df_train = pd.concat(Y_df_train, ignore_index=True)
Y_df_test = pd.concat(Y_df_test, ignore_index=True)
Y_df_validation = pd.concat(Y_df_validation, ignore_index=True)

Y_df = pd.concat([Y_df_train, Y_df_test, Y_df_validation], ignore_index=True)



n_time = len(Y_df.ds.unique())
val_size = int(.2 * n_time)
test_size = int(.2 * n_time)
#train_combined_dataX = np.concatenate(train_combined_dataX, axis=0)
#train_combined_dataY = np.concatenate(train_combined_dataY, axis=0)
#for patient in CLEANEDTEST_FILE_PATHS:
#    dataset_a, patientdata = load(patient)
#    patient_df = dataset_creator(dataset_a, 15, patientdata['id'])
#    dataX, dataY = create_variable_sliding_window_dataset_multiple_features(patient_df, 3, 15, i, targeti)
#    test_combined_dataX.append(dataX)
#    test_combined_dataY.append(dataY)
#test_combined_dataX = np.concatenate(test_combined_dataX, axis=0)
#test_combined_dataY = np.concatenate(test_combined_dataY, axis=0)
horizon = 96 # 24hrs = 4 * 15 min.
models = [Informer(h=horizon,                 # Forecasting horizon
                input_size=horizon,           # Input size
                max_steps=1000,               # Number of training iterations
                val_check_steps=100,          # Compute validation loss every 100 steps
                early_stop_patience_steps=3), # Stop training if validation loss does not improve
          Autoformer(h=horizon,
                input_size=horizon,
                max_steps=1000,
                val_check_steps=100,
                early_stop_patience_steps=3),
          PatchTST(h=horizon,
                input_size=horizon,
                max_steps=1000,
                val_check_steps=100,
                early_stop_patience_steps=3),
         ]

nf = NeuralForecast(
    models=models,
    freq='5min')

Y_hat_df = nf.cross_validation(df=Y_df,
                               val_size=val_size,
                               test_size=test_size,
                               n_windows=None)

Y_plot = Y_hat_df[Y_hat_df['unique_id']==559] # OT dataset
cutoffs = Y_hat_df['cutoff'].unique()[::horizon]
Y_plot = Y_plot[Y_hat_df['cutoff'].isin(cutoffs)]

plt.figure(figsize=(20,5))
plt.plot(Y_plot['ds'], Y_plot['y'], label='True')
plt.plot(Y_plot['ds'], Y_plot['Informer'], label='Informer')
plt.plot(Y_plot['ds'], Y_plot['Autoformer'], label='Autoformer')
plt.plot(Y_plot['ds'], Y_plot['PatchTST'], label='PatchTST')
plt.xlabel('Datestamp')
plt.ylabel('OT')
plt.grid()
plt.legend()
plt.show()


mae_informer = mae(Y_hat_df['y'], Y_hat_df['Informer'])
mae_autoformer = mae(Y_hat_df['y'], Y_hat_df['Autoformer'])
mae_patchtst = mae(Y_hat_df['y'], Y_hat_df['PatchTST'])

print(f'Informer: {mae_informer:.3f}')
print(f'Autoformer: {mae_autoformer:.3f}')
print(f'PatchTST: {mae_patchtst:.3f}')


threshold = 0.5
Y_hat_df['Informer_pred_class'] = (Y_hat_df['Informer'] >= threshold).astype(int)
Y_hat_df['Autoformer_pred_class'] = (Y_hat_df['Autoformer'] >= threshold).astype(int)
Y_hat_df['PatchTST_pred_class'] = (Y_hat_df['PatchTST'] >= threshold).astype(int)

print("NaN counts:")
print("y:", Y_hat_df['y'].isna().sum())
print("Informer_pred_class:", Y_hat_df['Informer_pred_class'].isna().sum())
print("Autoformer_pred_class:", Y_hat_df['Autoformer_pred_class'].isna().sum())
print("PatchTST_pred_class:", Y_hat_df['PatchTST_pred_class'].isna().sum())

# Drop rows where 'y' or predictions are NaN
Y_hat_df_clean = Y_hat_df.dropna(subset=['y', 'Informer_pred_class', 'Autoformer_pred_class', 'PatchTST_pred_class'])

# Ensure the data types are consistent for metric calculation
Y_hat_df_clean['y'] = Y_hat_df_clean['y'].astype(int)

# Calculate the metrics for Informer
# Ensure you're using the cleaned DataFrame for metric calculations
accuracy_informer = accuracy_score(Y_hat_df_clean['y'], Y_hat_df_clean['Informer_pred_class'])
f1_informer = f1_score(Y_hat_df_clean['y'], Y_hat_df_clean['Informer_pred_class'])
recall_informer = recall_score(Y_hat_df_clean['y'], Y_hat_df_clean['Informer_pred_class'])
precision_informer = precision_score(Y_hat_df_clean['y'], Y_hat_df_clean['Informer_pred_class'])

accuracy_autoformer = accuracy_score(Y_hat_df_clean['y'], Y_hat_df_clean['Autoformer_pred_class'])
f1_autoformer = f1_score(Y_hat_df_clean['y'], Y_hat_df_clean['Autoformer_pred_class'])
recall_autoformer = recall_score(Y_hat_df_clean['y'], Y_hat_df_clean['Autoformer_pred_class'])
precision_autoformer = precision_score(Y_hat_df_clean['y'], Y_hat_df_clean['Autoformer_pred_class'])

accuracy_patchtst = accuracy_score(Y_hat_df_clean['y'], Y_hat_df_clean['PatchTST_pred_class'])
f1_patchtst = f1_score(Y_hat_df_clean['y'], Y_hat_df_clean['PatchTST_pred_class'])
recall_patchtst = recall_score(Y_hat_df_clean['y'], Y_hat_df_clean['PatchTST_pred_class'])
precision_patchtst = precision_score(Y_hat_df_clean['y'], Y_hat_df_clean['PatchTST_pred_class'])

# Print the metrics for each model
print(f"Informer - Accuracy: {accuracy_informer:.3f}, F1 Score: {f1_informer:.3f}, Recall: {recall_informer:.3f}, Precision: {precision_informer:.3f}")
print(f"Autoformer - Accuracy: {accuracy_autoformer:.3f}, F1 Score: {f1_autoformer:.3f}, Recall: {recall_autoformer:.3f}, Precision: {precision_autoformer:.3f}")
print(f"PatchTST - Accuracy: {accuracy_patchtst:.3f}, F1 Score: {f1_patchtst:.3f}, Recall: {recall_patchtst:.3f}, Precision: {precision_patchtst:.3f}")


