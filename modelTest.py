import numpy as np
import pandas as pd
from scipy.linalg import svd
from xml_read import load
from defines import *
from datetime import timedelta

def carbohydrate_event_occurred(df, k,minutesCarb):

    time=df['glucose_level']['ts'][k]
    time_x_minutes_before = time - timedelta(minutes=minutesCarb)
    mask = (df['meal']['ts'] >= time_x_minutes_before) & (df['meal']['ts'] <= time)
    exists_between = (df['meal'][mask] != pd.Timestamp(0)).any()
    return exists_between['ts']


def exercise_event_occurred(df, k, minutesCarb):
    time = df['glucose_level']['ts'][k]
    time_x_minutes_before = time - timedelta(minutes=minutesCarb)

    mask = (df['exercise']['ts'] >= time_x_minutes_before) & (df['exercise']['ts'] <= time)

    # Checking if there's any timestamp between those times in the exercise column
    exists_between = (df['exercise'][mask] != pd.Timestamp(0)).any()
    return exists_between['ts']




def bolus_injection_occurred(df, k, minutesCarb):
    time = df['glucose_level']['ts'][k]
    time_x_minutes_before = time - timedelta(minutes=minutesCarb)

    mask = (df['bolus']['ts_end'] >= time_x_minutes_before) & (df['bolus']['ts_end'] <= time)

    # Checking if there's any timestamp between those times in the exercise column
    exists_between = (df['bolus'][mask] != pd.Timestamp(0)).any()
    return exists_between['ts_end']

def hankelize(data):
    L = len(data) // 2 + 1
    J = len(data) - L + 1
    H = np.zeros((L, J))
    for i in range(L):
        for j in range(J):
            H[i, j] = data.iloc[i + j]
    return H


def diagonal_averaging(mat):
    return np.mean(mat, axis=1)

def segment_data(data,outliers):
    segments = []

    # Starting point for the first segment
    start_idx = 0

    for outlier in outliers:
        # If there's a difference of more than one between the start index and the outlier,
        # add this segment to the list (this ensures we don't add empty segments).
        if outlier - start_idx > 1:
            segments.append(data[start_idx:outlier])

        # Set the starting index for the next segment to be the element after the current outlier.
        start_idx = outlier + 1

    # Handle the segment from the last outlier to the end of the data.
    if start_idx < len(data):
        segments.append(data[start_idx:])

    return segments

def process_cgm_values(train,minutesCarb,minutesBolus,minutesExercise):
    # Step 1


    train['glucose_level']['value']=train['glucose_level']['value'].values.astype(int)
    train['meal']['carbs'] = train['meal']['carbs'].values.astype(int)
    train['bolus']['dose'] = train['bolus']['dose'].values.astype(float)
    train['exercise']['intensity'] = train['exercise']['intensity'].values.astype(float)



    outliers = []
    for k in range(1, len( train['glucose_level']['value'])):
        diff =  train['glucose_level']['value'][k] -  train['glucose_level']['value'][k - 1]
        if abs(diff) > 30:
            outlier_condition = True
            if abs(diff) > 30 and carbohydrate_event_occurred(train, k,minutesCarb)==True:
                outlier_condition = False
            elif diff < 30 and bolus_injection_occurred(train, k,minutesBolus)==True:
                outlier_condition = False
            elif diff < 30 and exercise_event_occurred(train, k,minutesExercise)==True:
                outlier_condition = False
            if outlier_condition:
                outliers.append(k)

    segments = segment_data(train['glucose_level']['value'], outliers)

    denoised_segments = []

    for segment in segments:
        Q = segment

        H = hankelize(Q)
        U, S, Vt = svd(H)

        cumulative_sum = np.cumsum(S)
        total_sum = cumulative_sum[-1]
        indices_to_keep = np.where(cumulative_sum < 0.6 * total_sum)[0]
        S_reduced = np.diag(S[indices_to_keep])

        U_reduced = U[:, indices_to_keep]
        Vt_reduced = Vt[indices_to_keep, :]
        H_reconstructed = U_reduced @ S_reduced @ Vt_reduced
        denoised_segment = diagonal_averaging(H_reconstructed)

        denoised_segments.append(denoised_segment)
    highest_value = float('-inf')
    for arr in denoised_segments:
        current_max = np.max(arr)
        if current_max > highest_value:
            highest_value = current_max

    print(highest_value)
    return denoised_segments

data,_=load(TRAIN2_544_PATH)
test=process_cgm_values(data,45,30,30)
print("""""")

