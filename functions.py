import math
import ntpath
from datetime import datetime

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xml_read import *
from xml_write import *
from defines import *
import pandas as pd
import numpy as np
import xml.dom.minidom as minidom

from xml_write import filepath_to_string


def expand_peak(arr, expansion_factor=2, expansion_multiplier=0.8):
    expanded_arr = [0] * len(arr)

    for i in range(len(arr)):
        if arr[i] != 0:
            expanded_peak = [0] * (expansion_factor * 2 + 1)
            expanded_peak[expansion_factor] = arr[i]

            for j in range(expansion_factor):
                value = expanded_peak[expansion_factor + j] * expansion_multiplier
                expanded_peak[expansion_factor + j + 1] = value
                expanded_peak[expansion_factor - j - 1] = value

            for j in range(len(expanded_peak)):
                expanded_arr[i - expansion_factor + j] = expanded_peak[j]

    return expanded_arr



def dataPrepare(dataTrain, dataTest, backward_slidingwindow,forward_slidingwindow, maxfiltersize=10,oversampling=False):
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
    features_validation_y = ndimage.maximum_filter(features_validation_y, size=maxfiltersize)
    features_validation_y = pd.DataFrame(features_validation_y)

    features_validation_x = features_validation['value']
    features_validation_x = pd.DataFrame(features_validation_x)
    features_validation_x = features_validation_x.ffill()
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

    print("Shape of trainX:", featurestrain.shape)
    print("Shape of trainY:", featuresvalidation.shape)
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

def dataPrepareRegression(dataTrain, dataTest, backward_slidingwindow, forward_slidingwindow, oversampling=False, expansion_factor=2, expansion_multiplier=0.8,scaling=False):
    #TRAIN
    train_glucose_level_df = dataTrain['glucose_level']
    train_meal_df = dataTrain['meal']
    train_glucose_level_df['ts'] = pd.to_datetime(train_glucose_level_df['ts'])
    train_meal_df['ts'] = pd.to_datetime(train_meal_df['ts'])

    train_merged_df = pd.merge(train_glucose_level_df, train_meal_df, on='ts',how='outer')

    train_merged_df = train_merged_df.sort_values(by='ts')

    train_merged_df.reset_index(drop=True, inplace=True)
    train_merged_df.drop(columns=['type'], inplace=True)
    train_merged_df.drop(columns=['ts'], inplace=True)

    train_merged_df['carbs'].fillna(0, inplace=True)
    train_merged_df['value'] = pd.to_numeric(train_merged_df['value'], errors='coerce')
    train_merged_df['value'] = train_merged_df['value'].fillna((train_merged_df['value'].shift() + train_merged_df['value'].shift(-1)) / 2)

    train_merged_df['carbs']=train_merged_df['carbs'].values.astype(float)
    train_merged_df['value']=train_merged_df['value'].values.astype(float)

    train_merged_df['carbs'] = expand_peak(train_merged_df['carbs'],expansion_factor,expansion_multiplier)
    train_merged_df = pd.DataFrame(train_merged_df)

    # TEST
    test_glucose_level_df = dataTest['glucose_level']
    test_meal_df = dataTest['meal']
    test_glucose_level_df['ts'] = pd.to_datetime(test_glucose_level_df['ts'])
    test_meal_df['ts'] = pd.to_datetime(test_meal_df['ts'])

    test_merged_df = pd.merge(test_glucose_level_df, test_meal_df, on='ts', how='outer')

    test_merged_df = test_merged_df.sort_values(by='ts')

    test_merged_df.reset_index(drop=True, inplace=True)
    test_merged_df.drop(columns=['type'], inplace=True)
    test_merged_df.drop(columns=['ts'], inplace=True)

    test_merged_df['carbs'].fillna(0, inplace=True)
    test_merged_df['value'] = pd.to_numeric(test_merged_df['value'], errors='coerce')
    test_merged_df['value'] = test_merged_df['value'].fillna(
        (test_merged_df['value'].shift() + test_merged_df['value'].shift(-1)) / 2)

    test_merged_df['carbs'] = test_merged_df['carbs'].values.astype(float)
    test_merged_df['value'] = test_merged_df['value'].values.astype(float)

    test_merged_df['carbs'] = expand_peak(test_merged_df['carbs'], expansion_factor, expansion_multiplier)
    test_merged_df = pd.DataFrame(test_merged_df)

    test_merged_df = test_merged_df.dropna()
    train_merged_df = train_merged_df.dropna()


    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    if(scaling==True):
        train_merged_df['value'] = scaler_x.fit_transform(train_merged_df['value'].values.reshape(-1, 1)).flatten()


        test_merged_df['value'] = scaler_x.transform(test_merged_df['value'].values.reshape(-1, 1)).flatten()


        train_merged_df['carbs'] = scaler_y.fit_transform(train_merged_df['carbs'].values.reshape(-1, 1)).flatten()


        test_merged_df['carbs'] = scaler_y.transform(test_merged_df['carbs'].values.reshape(-1, 1)).flatten()

    print("Shape of trainX:", train_merged_df.shape)
    print("Shape of trainY:", test_merged_df.shape)
    train_merged_df=train_merged_df.values
    train_merged_df[:, [0, 1]] = train_merged_df[:, [1, 0]]
    test_merged_df=test_merged_df.values
    test_merged_df[:, [0, 1]] = test_merged_df[:, [1, 0]]

    trainX, trainY = create_variable_sliding_window_dataset(train_merged_df, backward_slidingwindow,
                                                            forward_slidingwindow)
    validX, validY = create_variable_sliding_window_dataset(test_merged_df, backward_slidingwindow,
                                                            forward_slidingwindow)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 1))

    print("Shape of trainX:", trainX.shape)
    print("Shape of trainY:", trainY.shape)
    print("Shape of validX:", validX.shape)
    print("Shape of validY:", validY.shape)

    return trainX, trainY, validX, validY



def count_missing_data(measurements: pd.DataFrame, time_step: pd.Timedelta) -> dict[str, int]:
    counts = {'good': 0,
              'missing': 0,
              'less_than_expected_time_delay': 0,
              'slightly_less_than_expected_time_delay': 0,
              'slightly_more_than_expected_time_delay': 0,
              'more_than_expected_time_delay': 0,
              'invalid_timestamp': 0
              }

    prev_ts = None
    for ts in measurements['ts']:
        if prev_ts is not None:
            dt = ts - prev_ts
            if ts is pd.NaT:
                counts['invalid_timestamp'] += 1
            elif dt == time_step:
                counts['good'] += 1
            elif dt < time_step or \
                    (time_step < dt < (time_step * 1.5)):

                if dt < (time_step - pd.Timedelta(seconds=10)):
                    counts['less_than_expected_time_delay'] += 1
                elif dt < time_step:
                    counts['slightly_less_than_expected_time_delay'] += 1
                elif time_step < dt <= (time_step + pd.Timedelta(seconds=1)):
                    counts['slightly_more_than_expected_time_delay'] += 1
                else:
                    counts['more_than_expected_time_delay'] += 1

            else:
                counts['missing'] += math.floor(dt.total_seconds() / time_step.total_seconds()) - 1
                counts['good'] += 1  # current time stamp is not missing duh
        else:
            counts['good'] += 1

        prev_ts = ts

    return counts

def loadeverycleanedxml():
    train_dicts = [load(f) for f in CLEANEDALL_TRAIN_FILE_PATHS]
    test_dicts = [load(f) for f in CLEANEDALL_TEST_FILE_PATHS]

    merged_train_data = {key: pd.DataFrame() for key in train_dicts[0][0].keys()}
    merged_test_data = {key: pd.DataFrame() for key in test_dicts[0][0].keys()}

    for train_dict in train_dicts:
        for key in merged_train_data.keys():
            if key in train_dict[0]:
                merged_train_data[key] = pd.concat([merged_train_data[key], train_dict[0][key]], ignore_index=True)

    for test_dict in test_dicts:
        for key in merged_test_data.keys():
            if key in test_dict[0]:
                merged_test_data[key] = pd.concat([merged_test_data[key], test_dict[0][key]], ignore_index=True)

    return merged_train_data,merged_test_data

def loadeveryxmlparam(train, test):
    train_dicts = [load(f) for f in train]
    test_dicts = [load(f) for f in test]

    merged_train_data = {key: pd.DataFrame() for key in train_dicts[0][0].keys()}
    merged_test_data = {key: pd.DataFrame() for key in test_dicts[0][0].keys()}

    for train_dict in train_dicts:
        for key in merged_train_data.keys():
            if key in train_dict[0]:
                merged_train_data[key] = pd.concat([merged_train_data[key], train_dict[0][key]], ignore_index=True)

    for test_dict in test_dicts:
        for key in merged_test_data.keys():
            if key in test_dict[0]:
                merged_test_data[key] = pd.concat([merged_test_data[key], test_dict[0][key]], ignore_index=True)

    return merged_train_data,merged_test_data

def loadeveryxml():
    train_dicts = [load(f) for f in ALL_TRAIN_FILE_PATHS]
    test_dicts = [load(f) for f in ALL_TEST_FILE_PATHS]

    merged_train_data = {key: pd.DataFrame() for key in train_dicts[0][0].keys()}
    merged_test_data = {key: pd.DataFrame() for key in test_dicts[0][0].keys()}

    for train_dict in train_dicts:
        for key in merged_train_data.keys():
            if key in train_dict[0]:
                merged_train_data[key] = pd.concat([merged_train_data[key], train_dict[0][key]], ignore_index=True)

    for test_dict in test_dicts:
        for key in merged_test_data.keys():
            if key in test_dict[0]:
                merged_test_data[key] = pd.concat([merged_test_data[key], test_dict[0][key]], ignore_index=True)

    return merged_train_data,merged_test_data
def missing_glucose_level_data(data):
    value=0
    current_day = pd.to_datetime(data['glucose_level']['ts'].iloc[0].date())
    last_day = pd.to_datetime(data['glucose_level']['ts'].iloc[-1].date())
    days=abs((last_day-current_day).days)+1
    glucose_value_max=days*288
    for glucoselevel in data['glucose_level']['ts']:
        value=value+1
    return value,glucose_value_max


def print_stats():
    for filepaths in ALL_FILE_PATHS:
        data, patient_data = load(filepaths)
        actual,max=missing_glucose_level_data(data)
        percentage=actual/max*100
        print(filepaths,': ',actual,'/',max,' Percentage:',percentage,'%')


def get_file_missing_glucose_data_statistics(data: dict[str, pd.DataFrame]):
    statistics_result = {}
    for data_type in data:
        time_step_found = False
        i = 0
        delta = None
        prev_time = None

        if data[data_type].columns.__contains__('ts'):
            for time in data[data_type]['ts']:
                if prev_time is not None:
                    delta = time - prev_time
                    if delta == pd.Timedelta(5, 'm'):
                        time_step_found = True
                        break

                prev_time = time

                i += 1

        if time_step_found:
            statistics_result[data_type] = count_missing_data(data[data_type], delta)

    return statistics_result


def drop_days_with_missing_glucose_data(data: dict[str, pd.DataFrame], missing_count_threshold) -> dict[
    str, pd.DataFrame]:
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()

    current_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[0].date())
    last_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[-1].date())

    while current_day <= last_day:
        next_day = current_day + pd.Timedelta(1, 'd')

        day = cleaned_data['glucose_level'][cleaned_data['glucose_level']['ts'] >= current_day]
        day = day[day['ts'] < next_day]

        if day.empty or get_file_missing_glucose_data_statistics({'glucose_level': day})['glucose_level'][
            'missing'] > missing_count_threshold:
            for measurement_type in cleaned_data.keys():
                # if any timestamp is in the day that is to be thrown away, throw away the entire event
                for measurement_parameter in cleaned_data[measurement_type]:
                    if types[measurement_type][measurement_parameter] == 'datetime':
                        tdf = cleaned_data[measurement_type]
                        day_data = tdf[tdf[measurement_parameter] >= current_day]
                        day_data = day_data[measurement_parameter][day_data[measurement_parameter] < next_day]

                        cleaned_data[measurement_type] = cleaned_data[measurement_type].drop(index=day_data.index)

        current_day = next_day

    return cleaned_data

def get_file_missing_eat_data_statistics(data: dict[str, pd.DataFrame]):
    statistics_result = {}
    for data_type in data:
        time_step_found = False
        i = 0
        delta = None
        prev_time = None

        if data[data_type].columns.__contains__('ts'):
            for time in data[data_type]['ts']:
                if prev_time is not None:
                    delta = time - prev_time
                    if delta == pd.Timedelta(5, 'm'):
                        time_step_found = True
                        break

                prev_time = time

                i += 1

        if time_step_found:
            statistics_result[data_type] = count_missing_data(data[data_type], delta)

    return statistics_result


def drop_days_with_missing_eat_data(data: dict[str, pd.DataFrame], missing_eat_threshold) -> dict[
    str, pd.DataFrame]:
    cleaned_data = {}
    #deepcopyzzuk
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    #kimentjük a napokat
    current_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[0].date())
    last_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[-1].date())
    #végig megyünk az összes napon
    if(cleaned_data['meal'].empty == False):
        while current_day <= last_day:
            next_day = current_day + pd.Timedelta(1, 'd')
            # kimentjük mindig az adott nap adatait a daybe
            day = cleaned_data['meal'][cleaned_data['meal']['ts'] >= current_day]
            day = day[day['ts'] < next_day]
            # ha a day üres, vagy kevesebb adat van benne mint a threshold akkor kuka
            if day.empty or len(day) < missing_eat_threshold:
                for measurement_type in cleaned_data.keys():
                    # if any timestamp is in the day that is to be thrown away, throw away the entire event
                    for measurement_parameter in cleaned_data[measurement_type]:
                        if types[measurement_type][measurement_parameter] == 'datetime':
                            tdf = cleaned_data[measurement_type]
                            day_data = tdf[tdf[measurement_parameter] >= current_day]
                            day_data = day_data[measurement_parameter][day_data[measurement_parameter] < next_day]

                            cleaned_data[measurement_type] = cleaned_data[measurement_type].drop(index=day_data.index)
            # váltunk a kövi napra
            current_day = next_day
        return cleaned_data
    else:
        for keys in cleaned_data.keys():
            cleaned_data[keys] = cleaned_data[keys].iloc[0:0]
        return cleaned_data

#region creating/inserting dataframes
def insert_row(idx, df, df_insert):
    if idx == -1:
        df = pd.concat([df, df_insert])
        df = df.reset_index(drop=True)
    else:
        dfA = df.iloc[:idx, ]
        dfB = df.iloc[idx:, ]

        df = pd.concat([dfA, df_insert, dfB])
        df = df.reset_index(drop=True)

    return df



def create_increasing_rows_fixed(amount, datetime, avg):
    rows = pd.DataFrame(index=np.arange(0, amount), columns=('ts', 'value'))
    next_datetime = datetime

    for i in range(0, amount):
        next_datetime += pd.Timedelta(5, 'm')
        dt = pd.to_datetime(next_datetime, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        rows.loc[i] = pd.Series({'ts': dt, 'value': math.floor(avg)})
    return rows



def create_decreasing_rows_fixed(amount, datetime, value):
    rows = pd.DataFrame(index=np.arange(0, amount), columns=('ts', 'value'))
    prev_datetime = datetime

    for i in range(0, amount):
        dt = pd.to_datetime(prev_datetime, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        rows.loc[i] = pd.Series({'ts': dt, 'value': math.floor(value)})
        prev_datetime += pd.Timedelta(5, 'm')
    return rows


def create_increasing_rows_continuous(amount, datetime,value_before_gap,value_after_gap):
    rows = pd.DataFrame(index=np.arange(0, amount), columns=('ts', 'value'))
    next_datetime = datetime
    value_before_gap=int(value_before_gap)
    value_after_gap_gap = int(value_after_gap)
    segment = (value_after_gap_gap - value_before_gap) / amount
    value = value_before_gap

    for i in range(0, amount):
        next_datetime += pd.Timedelta(5, 'm')
        dt = pd.to_datetime(next_datetime, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        rows.loc[i] = pd.Series({'ts': dt, 'value': value})
        value = int(value + segment)
    return rows


def create_decreasing_rows_continuous(amount, datetime,avg,valueaftergap):
    valueaftergap = int(valueaftergap)
    rows = pd.DataFrame(index=np.arange(0, amount), columns=('ts', 'value'))
    prev_datetime = datetime
    segment=(valueaftergap-avg)/amount
    value=int(avg)

    for i in range(0, amount):
        dt = pd.to_datetime(prev_datetime, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        rows.loc[i] = pd.Series({'ts': dt, 'value': value})
        prev_datetime += pd.Timedelta(5, 'm')
        value=int(value+segment)
    return rows

#endregion

#region Fill continous

def fill_start_glucose_level_data_continuous(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    avgs = avg_calculator(data)
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()

    # mindenekelőtt megnézzük hogy az első elem 0 óra környékén van e
    first_timestamp = pd.Timestamp(year=cleaned_data['glucose_level']['ts'][0].year,
                                   month=cleaned_data['glucose_level']['ts'][0].month,
                                   day=cleaned_data['glucose_level']['ts'][0].day,
                                   hour=cleaned_data['glucose_level']['ts'][0].hour,
                                   minute=cleaned_data['glucose_level']['ts'][0].minute,
                                   second=cleaned_data['glucose_level']['ts'][0].second)
    # kimentjük a 0 órát egy változóba
    hour_zero = pd.Timestamp(year=cleaned_data['glucose_level']['ts'][0].year,
                             month=cleaned_data['glucose_level']['ts'][0].month,
                             day=cleaned_data['glucose_level']['ts'][0].day,
                             hour=0, minute=0, second=0)
    # megnézzük a különbséget
    first_amount = first_timestamp - hour_zero
    if first_amount > pd.Timedelta(10, 'm'):
        # megnézzük mennyi elem hiányzik
        first_amount_missing = math.floor(first_amount.total_seconds() / time_step.total_seconds()) - 1
        df_to_insert = create_decreasing_rows_continuous(first_amount_missing, hour_zero, int(avgs[0]),cleaned_data['glucose_level']['value'][0])
        cleaned_data['glucose_level'] = insert_row(0, cleaned_data['glucose_level'], df_to_insert)

    return cleaned_data
    ######################################




def fill_glucose_level_data_continuous(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    cleaned_data = fill_start_glucose_level_data_continuous(cleaned_data, time_step)
    avg_index = 0
    prev_ts = None
    #mivel a ciklusban nem frissül az indexelés, azaz a régi adatbázison fut végig, kell 1 korrekció
    #amivel a beszúrást oldjuk meg. (ha beszúrunk 5 elemet a 65 index után, a 66. elem a régi 66. elem lesz
    # nem pedig az új beszúrt)
    corrector = 0
    for idx, ts in enumerate(cleaned_data['glucose_level']['ts']):
        if prev_ts is not None:
            dt = ts - prev_ts
            if ts.day != prev_ts.day:
                avg_index += 1
            if pd.Timedelta(1,'d') > dt >= time_step + time_step:
                # megnézzük mennyi hiányzik
                missing_amount = math.floor(dt.total_seconds() / time_step.total_seconds()) - 1

                # létrehozunk 1 dataframe-t amiben megfelelő mennyiségű sor van
                df_to_insert = create_increasing_rows_continuous(missing_amount, prev_ts,
                                                          cleaned_data['glucose_level']['value'][idx+corrector-1],cleaned_data['glucose_level']['value'][idx+corrector+1])
                # beszúrjuk az új dataframeünket az eredetibe
                cleaned_data['glucose_level'] = insert_row(idx+corrector, cleaned_data['glucose_level'], df_to_insert)
                corrector += missing_amount
        prev_ts = ts
    return cleaned_data

#endregion

# region Fill fixed
def fill_start_glucose_level_data_fixed(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    avgs = avg_calculator(data)
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()

    # mindenekelőtt megnézzük hogy az első elem 0 óra környékén van e
    first_timestamp = pd.Timestamp(year=data['glucose_level']['ts'][0].year,
                                   month=data['glucose_level']['ts'][0].month,
                                   day=data['glucose_level']['ts'][0].day,
                                   hour=data['glucose_level']['ts'][0].hour,
                                   minute=data['glucose_level']['ts'][0].minute,
                                   second=data['glucose_level']['ts'][0].second)
    # kimentjük a 0 órát egy változóba
    hour_zero = pd.Timestamp(year=data['glucose_level']['ts'][0].year,
                             month=data['glucose_level']['ts'][0].month,
                             day=data['glucose_level']['ts'][0].day,
                             hour=0, minute=0, second=0)
    # megnézzük a különbséget
    first_amount = first_timestamp - hour_zero
    if first_amount > pd.Timedelta(10, 'm'):
        # megnézzük mennyi elem hiányzik
        first_amount_missing = math.floor(first_amount.total_seconds() / time_step.total_seconds()) - 1
        df_to_insert = create_decreasing_rows_fixed(first_amount_missing, hour_zero, avgs[0])
        data['glucose_level'] = insert_row(0, data['glucose_level'], df_to_insert)

    return cleaned_data
    ######################################

def fill_glucose_level_data_fixed(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    avgs = avg_calculator(data)
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    cleaned_data = fill_start_glucose_level_data_fixed(cleaned_data, time_step)
    avg_index = 0
    prev_ts = None
    #mivel a ciklusban nem frissül az indexelés, azaz a régi adatbázison fut végig, kell 1 korrekció
    #amivel a beszúrást oldjuk meg. (ha beszúrunk 5 elemet a 65 index után, a 66. elem a régi 66. elem lesz
    # nem pedig az új beszúrt)
    corrector = 0
    for idx, ts in enumerate(data['glucose_level']['ts']):
        if prev_ts is not None:
            dt = ts - prev_ts
            if ts.day != prev_ts.day:
                avg_index += 1
            if pd.Timedelta(1,'d') > dt >= time_step + time_step:
                # megnézzük mennyi hiányzik
                missing_amount = math.floor(dt.total_seconds() / time_step.total_seconds()) - 1

                # létrehozunk 1 dataframe-t amiben megfelelő mennyiségű sor van
                df_to_insert = create_increasing_rows_fixed(missing_amount, prev_ts,
                                                          avgs[avg_index])
                # beszúrjuk az új dataframeünket az eredetibe
                cleaned_data['glucose_level'] = insert_row(idx+corrector, cleaned_data['glucose_level'], df_to_insert)
                corrector += missing_amount
        prev_ts = ts
    return cleaned_data


# endregion

def avg_calculator(data: dict[str, pd.DataFrame]):
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    glucose_level_average = []
    cleaned_data['glucose_level']['value'] = cleaned_data['glucose_level']['value'].astype(int)
    current_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[0].date())
    last_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[-1].date())
    while current_day <= last_day:

        # Van 1 nem létező napunk 544-esben 2027-05-24 nap missing (: ezért NaN-t ad vissza...
        next_day = current_day + pd.Timedelta(1, 'd')

        glucose_level_count = cleaned_data['glucose_level'][cleaned_data['glucose_level']['ts'] >= current_day]
        glucose_level_count = glucose_level_count[glucose_level_count['ts'] < next_day]
        average = np.sum(glucose_level_count['value']) / glucose_level_count.shape[0]
        if not math.isnan(float(average)):
            glucose_level_average.append(math.floor(average))
        current_day = next_day
    return glucose_level_average

#region Write to File
def write_all_cleaned_xml_fixed():
    for filepaths in ALL_FILE_PATHS:
        data, patient_data = load(filepaths)
        stringpath = filepath_to_string(filepaths)
        filled_data = fill_start_glucose_level_data_fixed(data, pd.Timedelta(5, 'm'))
        filled_data = fill_glucose_level_data_fixed(filled_data, pd.Timedelta(5, 'm'))
        write_to_xml(os.path.join(CLEANED_DATA_DIR2, stringpath), filled_data, int(patient_data['id']),patient_data['insulin_type'],body_weight=int(patient_data['weight']))


def write_all_cleaned_xml_continuous():
    for filepaths in ALL_FILE_PATHS:
        data, patient_data = load(filepaths)
        stringpath = filepath_to_string(filepaths)
        filled_data = fill_start_glucose_level_data_continuous(data, pd.Timedelta(5, 'm'))
        filled_data = fill_glucose_level_data_continuous(filled_data, pd.Timedelta(5, 'm'))
        write_to_xml(os.path.join(CLEANED_DATA_DIR2, stringpath), filled_data, int(patient_data['id']),patient_data['insulin_type'],body_weight=int(patient_data['weight']))

def write_all_cleaned_xml():
    for filepaths in ALL_FILE_PATHS:
        data, patient_data = load(filepaths)
        stringpath = filepath_to_string(filepaths)
        filled_data = data_cleaner(data, pd.Timedelta(5, "m"), 30, 3)
        write_to_xml(os.path.join(CLEANED_DATA_DIR2, stringpath), filled_data, int(patient_data['id']),patient_data['insulin_type'],body_weight=int(patient_data['weight']))

#endregion

#region Statistics
def count_glucose_level_data(threshholdnumber: int,mealcount: int):
    #print("Glucose level threshold number: {} | Meal threshold number: {}".format(threshholdnumber,mealcount))
    root="<root>Glucose level threshold number: {} | Meal threshold number: {}".format(threshholdnumber,mealcount)
    for filepaths in ALL_FILE_PATHS:
        data, patient_data = load(filepaths)
        stringpath = filepath_to_string(filepaths)
        cleaned_data = drop_days_with_missing_glucose_data(data, threshholdnumber)
        cleaned_data = drop_days_with_missing_eat_data(cleaned_data, mealcount)
        for key in cleaned_data.keys():
            cleaned_data[key] = cleaned_data[key].reset_index(drop=True)
        #print("{} Glucose level data amount: {} / {} ".format(stringpath,len(cleaned_data['glucose_level']),len(data['glucose_level'])))
        root+="<child>{} : Glucose level data amount: {} / {}</child>".format(stringpath,len(cleaned_data['glucose_level']),len(data['glucose_level']))
    root+="</root>"

    #tree.write("Patients.xml", encoding="utf-8", xml_declaration=True, method="xml",pretty_print=True)
    dom = minidom.parseString(root)
    pretty_xml_str = dom.toprettyxml()
    filename="Patients{}-{}.xml".format(threshholdnumber,mealcount)
    with open(filename, "w") as f:
        f.write(pretty_xml_str)


def count_glucose_windows(threshholdnumber: int):
    # print("Glucose level threshold number: {} | Meal threshold number: {}".format(threshholdnumber,mealcount))
    root = "<root>Glucose window size: {} ".format(threshholdnumber)
    for filepaths in ALL_FILE_PATHS:
        data, patient_data = load(filepaths)
        stringpath = filepath_to_string(filepaths)
        for key in data.keys():
            data[key] = data[key].reset_index(drop=True)
        # print("{} Glucose level data amount: {} / {} ".format(stringpath,len(cleaned_data['glucose_level']),len(data['glucose_level'])))
        root += "<child>{} : Glucose level data amount: {} </child>".format(stringpath,
                                                                            count_continuous_glucose_windows(
                                                                                data['glucose_level'],
                                                                                threshholdnumber))
    root += "</root>"

    # tree.write("Patients.xml", encoding="utf-8", xml_declaration=True, method="xml",pretty_print=True)
    dom = minidom.parseString(root)
    pretty_xml_str = dom.toprettyxml()
    filename = "Patients_windows{}.xml".format(threshholdnumber)
    with open(filename, "w") as f:
        f.write(pretty_xml_str)


def count_continuous_glucose_windows(data, thresholdglucose: int, time_step=pd.Timedelta(5, 'm')):
    counter = 0
    count = 0

    prev_ts = None
    for ts in data['ts']:
        if prev_ts is not None:
            dt = ts - prev_ts
            if ts is pd.NaT:
                counter = 0
            elif dt == time_step:
                counter += 1
                if (counter == thresholdglucose):
                    counter = 0
                    count += 1
            elif dt < time_step or \
                    (time_step < dt < (time_step * 1.5)):

                if dt < (time_step - pd.Timedelta(seconds=10)):
                    counter = 0

                elif dt < time_step:
                    counter += 1
                    if (counter == thresholdglucose):
                        counter = 0
                        count += 1
                elif time_step < dt <= (time_step + pd.Timedelta(seconds=1)):
                    counter += 1
                    if (counter == thresholdglucose):
                        counter = 0
                        count += 1
                else:
                    counter = 0
        else:
            counter += 1

        prev_ts = ts

    return count

def count_continuous_glucose_windows_overlapping(data, thresholdglucose: int, time_step=pd.Timedelta(5, 'm')):
    data_filled = {}
    for key in data.keys():
        data_filled[key] = data[key].__deepcopy__()
    data_filled = fill_glucose_level_data_with_zeros(data_filled, time_step)
    counter = 0
    idx = 0
    while idx < len(data_filled['glucose_level']['value'])-thresholdglucose:
        tmp = data_filled['glucose_level']['value'][idx:idx+thresholdglucose]
        if tmp.isin([0]).any():
            idx = tmp.loc[tmp == 0].index[-1]+1
        else:
            counter += 1
            idx += 1
    return counter

def count_glucose_level_data_overlapping(threshholdnumber: int):
    #print("Glucose level threshold number: {} | Meal threshold number: {}".format(threshholdnumber,mealcount))
    root="<root>Glucose level threshold number: {}".format(threshholdnumber)
    for filepaths in ALL_FILE_PATHS:
        data, patient_data = load(filepaths)
        stringpath = filepath_to_string(filepaths)
        cleaned_data = {}
        for key in data.keys():
            cleaned_data[key] = data[key].__deepcopy__()
        #print("{} Glucose level data amount: {} / {} ".format(stringpath,len(cleaned_data['glucose_level']),len(data['glucose_level'])))
        root+="<child>{} : Glucose level data amount: {}</child>".format(stringpath,count_continuous_glucose_windows_overlapping(data,threshholdnumber))
    root+="</root>"

    #tree.write("Patients.xml", encoding="utf-8", xml_declaration=True, method="xml",pretty_print=True)
    dom = minidom.parseString(root)
    pretty_xml_str = dom.toprettyxml()
    filename="Patients{}_overlapping.xml".format(threshholdnumber)
    with open(filename, "w") as f:
        f.write(pretty_xml_str)

def get_data_without_zeros(data):
    #ez csak akkor használható, ha már a meal is fel van töltve 0-ákkal a dataframeben
    data_sorted = {}
    for key in data.keys():
        data_sorted[key] = data[key].__deepcopy__()

    selected_pairs = ['meal', 'glucose_level']

    # Initialize an empty list to store the extracted DataFrames
    extracted_dataframes = []

    # Iterate over the dictionary items
    for key, df in data_sorted.items():
        if key in selected_pairs:
            # Extract the necessary columns from the DataFrame
            if key == 'meal':
                extracted_df = df[['carb']]
            elif key == 'glucose_level':
                extracted_df = df[['value']]
            extracted_dataframes.append(extracted_df)

    # Concatenate the extracted DataFrames into a single DataFrame
    result_df = pd.concat(extracted_dataframes, ignore_index=True)

    non_zero_rows = result_df[result_df['value'] != 0]
    non_zero_rows['group'] = (non_zero_rows['value'] == 0).cumsum()
    non_zero_rows_with_target = non_zero_rows.groupby('group').agg(
        {'value': list, 'carb': lambda x: list(x)}).reset_index(drop=True)

    return non_zero_rows_with_target

#endregion

#region Fill with zeros
#region Glucose
def fill_glucose_level_data_with_zeros_start(data, time_step):
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()

    # mindenekelőtt megnézzük hogy az első elem 0 óra környékén van e
    first_timestamp = pd.Timestamp(year=data['glucose_level']['ts'][0].year,
                                   month=data['glucose_level']['ts'][0].month,
                                   day=data['glucose_level']['ts'][0].day,
                                   hour=data['glucose_level']['ts'][0].hour,
                                   minute=data['glucose_level']['ts'][0].minute,
                                   second=data['glucose_level']['ts'][0].second)
    # kimentjük a 0 órát egy változóba
    hour_zero = pd.Timestamp(year=data['glucose_level']['ts'][0].year,
                             month=data['glucose_level']['ts'][0].month,
                             day=data['glucose_level']['ts'][0].day,
                             hour=0, minute=0, second=0)
    # megnézzük a különbséget
    first_amount = first_timestamp - hour_zero
    if first_amount > pd.Timedelta(10, 'm'):
        # megnézzük mennyi elem hiányzik
        first_amount_missing = math.floor(first_amount.total_seconds() / time_step.total_seconds()) - 1
        df_to_insert = create_decreasing_rows_fixed(first_amount_missing, hour_zero, 0)
        cleaned_data['glucose_level'] = insert_row(0, cleaned_data['glucose_level'], df_to_insert)

    return cleaned_data

def fill_glucose_level_data_zeros_end(data, time_step):
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()

    # kimentjük az utolsó glükóz órát is
    first_timestamp = pd.Timestamp(year=cleaned_data['glucose_level']['ts'].iloc[-1].year,
                                   month=cleaned_data['glucose_level']['ts'].iloc[-1].month,
                                   day=cleaned_data['glucose_level']['ts'].iloc[-1].day,
                                   hour=cleaned_data['glucose_level']['ts'].iloc[-1].hour,
                                   minute=cleaned_data['glucose_level']['ts'].iloc[-1].minute,
                                   second=cleaned_data['glucose_level']['ts'].iloc[-1].second)
    # kimentjük a nap végét egy változóba
    last_timestamp = pd.Timestamp(year=cleaned_data['glucose_level']['ts'].iloc[-1].year,
                                  month=cleaned_data['glucose_level']['ts'].iloc[-1].month,
                                  day=cleaned_data['glucose_level']['ts'].iloc[-1].day,
                                  hour=23, minute=59, second=00)
    # megnézzük a különbséget
    amount = last_timestamp - first_timestamp
    if amount > pd.Timedelta(10, 'm'):
        # megnézzük mennyi elem hiányzik
        first_amount_missing = math.floor(amount.total_seconds() / time_step.total_seconds()) - 1
        df_to_insert = create_increasing_rows_fixed(first_amount_missing, first_timestamp, 0)
        cleaned_data['glucose_level'] = insert_row(-1, cleaned_data['glucose_level'], df_to_insert)

    return cleaned_data

def fill_glucose_level_data_with_zeros(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    #ennek a lényege az, hogy először meghívja a startosat, amelyik az első adatot megvizsgálja és ha van még "hely",
    #hogy legyen éjfélig adat előtte akkor beszúr az első adat elé megfelelő számú adatsor 0 value-val

    #utána feltölti a köztes részeknél is 0-val a sorokat, majd az utolsó elemnél megnézi, hogy az utolsó elem után fér-e
    #még adatsor 23:59-ig, ha igen feltölti azokat is, <-- erre azért van szükség mert lehet aznap utána még van meal adat
    #és a meal fillje errora futna, ha nem tennénk

    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    prev_ts = None

    cleaned_data = fill_glucose_level_data_with_zeros_start(cleaned_data, time_step)
    #mivel a ciklusban nem frissül az indexelés, azaz a régi adatbázison fut végig, kell 1 korrekció
    #amivel a beszúrást oldjuk meg. (ha beszúrunk 5 elemet a 65 index után, a 66. elem a régi 66. elem lesz
    # nem pedig az új beszúrt)
    corrector = 0
    for idx, ts in enumerate(cleaned_data['glucose_level']['ts']):
        if prev_ts is not None:
            dt = ts - prev_ts
            if dt >= time_step + time_step:
                # megnézzük mennyi hiányzik
                missing_amount = math.floor(dt.total_seconds() / time_step.total_seconds()) - 1

                # létrehozunk 1 dataframe-t amiben megfelelő mennyiségű sor van
                df_to_insert = create_increasing_rows_fixed(missing_amount, prev_ts, 0)

                # beszúrjuk az új dataframeünket az eredetibe
                cleaned_data['glucose_level'] = insert_row(idx+corrector, cleaned_data['glucose_level'], df_to_insert)
                corrector += missing_amount
        prev_ts = ts
    cleaned_data = fill_glucose_level_data_zeros_end(cleaned_data, time_step)
    return cleaned_data
#endregion
#region Meal
def create_increasing_rows_meal(amount,  avg):
    rows = pd.DataFrame(index=np.arange(0, amount), columns=['carbs'])


    for i in range(0, amount):
        rows.loc[i] = pd.Series({'carbs': math.floor(avg)})
    return rows
def fill_end_meal_with_zeros(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()

    #kimentjük az utolsó meal órát is
    first_timestamp = pd.Timestamp(year=cleaned_data['meal']['ts'].iloc[-1].year,
                                   month=cleaned_data['meal']['ts'].iloc[-1].month,
                                   day=cleaned_data['meal']['ts'].iloc[-1].day,
                                   hour=cleaned_data['meal']['ts'].iloc[-1].hour,
                                   minute=cleaned_data['meal']['ts'].iloc[-1].minute,
                                   second=cleaned_data['meal']['ts'].iloc[-1].second)
    # kimentjük az utolsó glükóz órát egy változóba
    last_timestamp = pd.Timestamp(year=cleaned_data['glucose_level']['ts'].iloc[-1].year,
                             month=cleaned_data['glucose_level']['ts'].iloc[-1].month,
                             day=cleaned_data['glucose_level']['ts'].iloc[-1].day,
                             hour=cleaned_data['glucose_level']['ts'].iloc[-1].hour,
                             minute=cleaned_data['glucose_level']['ts'].iloc[-1].minute,
                             second=cleaned_data['glucose_level']['ts'].iloc[-1].second)
    # megnézzük a különbséget
    amount = last_timestamp - first_timestamp
    if amount > pd.Timedelta(10, 'm'):
        # megnézzük mennyi elem hiányzik
        first_amount_missing = math.floor(amount.total_seconds() / time_step.total_seconds()) - 1
        df_to_insert = create_increasing_rows_meal(first_amount_missing, 0)
        cleaned_data['meal'] = insert_row(-1, cleaned_data['meal'], df_to_insert)

    return cleaned_data
def fill_meal_with_zeros(data):

    prev_match_idx = 0
    corrector = 0
    for idx in range(len(data['meal'])):
        timestamp_to_match = data['meal']['ts'].iloc[idx+corrector]

        # Step 2: Define the time window
        window_start = timestamp_to_match - pd.Timedelta(minutes=5)
        window_end = timestamp_to_match + pd.Timedelta(minutes=5)

        # Step 3: Find the index of the first row in df1 that falls within the time window
        matching_index = data['glucose_level'].index[(data['glucose_level']['ts'] >= window_start) & (data['glucose_level']['ts'] <= window_end)].tolist()[0]
        #print(matching_index)
        #if matching_index == 2451:
        #    print(')')
        #if int((idx+corrector)/1000) == 7:
        #    print('!')
        amount = matching_index- prev_match_idx
        prev_match_idx = matching_index+1
        df_to_insert = create_increasing_rows_meal(amount, 0)

        # beszúrjuk az új dataframeünket az eredetibe
        data['meal'] = insert_row(idx + corrector, data['meal'], df_to_insert)
        if amount > 0:
            corrector += amount
    data = fill_end_meal_with_zeros(data,pd.Timedelta(minutes=5))
    return data
#endregion
#endregion



def load_everything():
    train_dict = {}
    file = open('MealDataCompare.txt','w')
    file.write('Train ')
    print('Train')
    idx = 0
    for filepaths in ALL_TRAIN_FILE_PATHS:
        #print(filepaths[-19:-4])
        to_write= filepaths[-19:-16] + '\n'
        file.write(to_write)
        #betöltjük egyesével az xml fájlokat
        temp_data, temp_patient_data = load(filepaths)
        #print('Glükóz adatok: ',len(temp_data['glucose_level']), '\nMeal adatok: ', len(temp_data['meal']))

        #mindegyiknél feltöltjük 0-val a hiányzó részeket, hogy "létezzenek"
        temp_data = fill_glucose_level_data_with_zeros(temp_data, pd.Timedelta(minutes=5))
        #print('Glükóz adatok: ', len(temp_data['glucose_level']), '\nMeal adatok: ', len(temp_data['meal']))

        #kidobjuk azokat a meal adatokat, amelyeknél nem létezik egyező napú glükóz adat
        temp_data = drop_meal_days(temp_data)

        #feltöltjük a mealt is 0-ákkal, hogy létezzenek, a glükóz adatoknak megfelelően
        temp_data = fill_meal_with_zeros(temp_data)
        #print('Glükóz adatok: ', len(temp_data['glucose_level']), '\nMeal adatok: ', len(temp_data['meal']))

        to_write = 'Glucose adatok: ' + str(len(temp_data['glucose_level'])) + '\nMeal adatok: '+ str(len(temp_data['meal'])) + '\n\n'
        file.write(to_write)
        kulonbseg = len(temp_data['glucose_level']) - len(temp_data['meal'])
        temp_data['glucose_level'] = temp_data['glucose_level'][0:-kulonbseg]
        #majd itt egybefűzzük az egészet egy nagy dataframe-be
        for key, value in temp_data.items():
            if key in train_dict:
                train_dict[key] = pd.concat([train_dict[key], value])
                train_dict[key] = train_dict[key].reset_index(drop=True)
            else:
                train_dict[key] = value
                train_dict[key] = train_dict[key].reset_index(drop=True)
        idx += 1
       #if idx == 1:
       #    break

    test_dict = {}
    file.write('Test ')
    print('Test')
    idx = 0
    for filepaths in ALL_TEST_FILE_PATHS:
        #ugyanaz történik csak a test fájlokkal
        #print(filepaths[-18:-4])
        to_write = filepaths[-18:-15] + '\n'
        file.write(to_write)
        temp_data, temp_patient_data = load(filepaths)
        #print('Glükóz adatok: ', len(temp_data['glucose_level']), '\nMeal adatok: ', len(temp_data['meal']))

        temp_data = fill_glucose_level_data_with_zeros(temp_data, pd.Timedelta(minutes=5))
        #print('Glükóz adatok: ', len(temp_data['glucose_level']), '\nMeal adatok: ', len(temp_data['meal']))

        temp_data = drop_meal_days(temp_data)

        temp_data = fill_meal_with_zeros(temp_data)
        #print('Glükóz adatok: ', len(temp_data['glucose_level']), '\nMeal adatok: ', len(temp_data['meal']))
        to_write = 'Glucose adatok: ' + str(len(temp_data['glucose_level'])) + '\nMeal adatok: '+ str(len(temp_data['meal'])) + '\n\n'
        file.write(to_write)
        kulonbseg = len(temp_data['glucose_level']) - len(temp_data['meal'])
        temp_data['glucose_level'] = temp_data['glucose_level'][0:-kulonbseg]
        for key, value in temp_data.items():
            if key in test_dict:
                test_dict[key] = pd.concat([test_dict[key], value])
                test_dict[key] = test_dict[key].reset_index(drop=True)
            else:
                test_dict[key] = value
                test_dict[key] = test_dict[key].reset_index(drop=True)
        idx += 1
        #if idx == 1:
        #    break
    file.close()
    return train_dict, test_dict



def drop_meal_days(data):
    # Step 1: Extract the unique dates from the "glucose level" dataset
    glucose_dates = pd.to_datetime(data['glucose_level']['ts']).dt.date.unique()
    #print('Meal hossza: ', len(data['meal']))
    # Step 2: Filter the "meal" dataset to keep only the rows where the date exists in the "glucose level" dataset
    data['meal'] = data['meal'][pd.to_datetime(data['meal']['ts']).dt.date.isin(glucose_dates)]
    #print('Törlés után: ', len(data['meal']))

    return data

def train_test_valid_split(glucose_data: pd.DataFrame):
    cleaned_data = {}
    for key in glucose_data.keys():
        cleaned_data[key] = glucose_data[key].__deepcopy__()
    cleaned_data = pd.DataFrame(cleaned_data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    cleaned_data = scaler.fit_transform(cleaned_data)
    idx = int(0.8 * int(cleaned_data.shape[0]))
    train_x = cleaned_data[:idx]
    test_x = cleaned_data[idx:]
    return train_x,  test_x

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 1]
        dataX.append(a)
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)

def create_dataset_multi(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 2]
        dataX.append(a)
        dataY.append(dataset[i, 0:2])
    return np.array(dataX), np.array(dataY)
def create_variable_sliding_window_dataset(dataset, backward_steps, forward_steps):
    if not isinstance(dataset, np.ndarray):
        dataset = dataset.values
    dataX, dataY = [], []
    #assumes second column is the target
    for i in range(backward_steps, len(dataset) - forward_steps):
        a = dataset[i-backward_steps:(i +forward_steps), 1]
        b = dataset[i, 0]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


def write_model_stats_out_xml_classification(history, train, prediction, filename, backward_slidingwindow, forward_slidingwindow, maxfiltersize, learning_rate, oversampling):
    #print("Glucose level threshold number: {} | Meal threshold number: {}".format(threshholdnumber,mealcount))
    root = "<root>"
    root += "<model>Model details:"
    root += "<hyperparameters>"
    root += "<backward_slidingwindow> {} </backward_slidingwindow>".format(backward_slidingwindow)
    root += "<forward_slidingwindow> {} </forward_slidingwindow>".format(forward_slidingwindow)
    root += "<maxfiltersize> {} </maxfiltersize>".format(maxfiltersize)
    root += "<learning_rate> {} </learning_rate>".format(learning_rate)
    root += "<oversampling> {} </oversampling>".format(oversampling)
    root += "</hyperparameters>"
    root += "<layers>"
    layer = history.model.layers
    for idx in range(len(history.model.layers)):
        if "dropout" in layer[idx].name:
            root += "<layer>{} name:{} units:{} </layer>".format(idx+1, layer[idx].name, layer[idx].rate)
        elif "input" in layer[idx].name:
            root += "<layer>{} name:{} shape:{} </layer>".format(idx+1, layer[idx].name, layer[idx].input_shape)
        elif "dense" in layer[idx].name:
            root += "<layer>{} name:{} units:{} </layer>".format(idx + 1, layer[idx].name,
                                                                               layer[idx].units,)
        else:
            root += "<layer>{} name:{} units:{}  return_sequences:{} </layer>".format(idx+1,layer[idx].name, layer[idx].units, layer[idx].return_sequences)
    root += "</layers>"
    root += "<history>"
    loss = history.history["loss"]
    precision = history.history["precision"]
    accuracy = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_precision = history.history["val_precision"]
    val_accuracy = history.history["val_accuracy"]
    for idx in range(len(history.history["loss"])):
        root += "<metrics> epoch:{} loss:{} precision:{}, accuracy:{}, val_loss:{}, val_precision:{}, val_accuracy:{}</metrics>".format(idx, loss[idx], precision[idx], accuracy[idx], val_loss[idx], val_precision[idx], val_accuracy[idx])
    root += "</history>"
    root += "</model>"
    root += "<data>"
    for i in range (len(train)):
        root += "<row> train/prediction: {}/{} </row>".format(train[i],prediction[i])

    root += "</data>"
    root += "</root>"
    #tree.write("Patients.xml", encoding="utf-8", xml_declaration=True, method="xml",pretty_print=True)
    dom = minidom.parseString(root)
    pretty_xml_str = dom.toprettyxml()
    now = datetime.now()
    dt_string = now.strftime("%Y.%m.%d_%H.%M")
    filename = "{}_{}.xml".format(filename, dt_string)
    with open(filename, "w") as f:
        f.write(pretty_xml_str)

def write_model_stats_out_xml_regression(history, train, prediction, filename, backward_slidingwindow, forward_slidingwindow, maxfiltersize, learning_rate, oversampling):
    #print("Glucose level threshold number: {} | Meal threshold number: {}".format(threshholdnumber,mealcount))
    root = "<root>"
    root += "<model>Model details:"
    root += "<hyperparameters>"
    root += "<backward_slidingwindow> {} </backward_slidingwindow>".format(backward_slidingwindow)
    root += "<forward_slidingwindow> {} </forward_slidingwindow>".format(forward_slidingwindow)
    root += "<maxfiltersize> {} </maxfiltersize>".format(maxfiltersize)
    root += "<learning_rate> {} </learning_rate>".format(learning_rate)
    root += "<oversampling> {} </oversampling>".format(oversampling)
    root += "</hyperparameters>"
    root += "<layers>"
    layer = history.model.layers
    for idx in range(len(history.model.layers)):
        if "dropout" in layer[idx].name:
            root += "<layer>{} name:{} units:{} </layer>".format(idx+1, layer[idx].name, layer[idx].rate)
        elif "input" in layer[idx].name:
            root += "<layer>{} name:{} shape:{} </layer>".format(idx+1, layer[idx].name, layer[idx].input_shape)
        elif "dense" in layer[idx].name:
            root += "<layer>{} name:{} units:{} </layer>".format(idx + 1, layer[idx].name,
                                                                               layer[idx].units,)
        else:
            root += "<layer>{} name:{} units:{}  return_sequences:{} </layer>".format(idx+1,layer[idx].name, layer[idx].units, layer[idx].return_sequences)
    root += "</layers>"
    root += "<history>"
    loss = history.history["loss"]
    precision = history.history["precision"]
    accuracy = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_precision = history.history["val_precision"]
    val_accuracy = history.history["val_accuracy"]
    for idx in range(len(history.history["loss"])):
        root += "<metrics> epoch:{} loss:{} precision:{}, accuracy:{}, val_loss:{}, val_precision:{}, val_accuracy:{}</metrics>".format(idx, loss[idx], precision[idx], accuracy[idx], val_loss[idx], val_precision[idx], val_accuracy[idx])
    root += "</history>"
    root += "</model>"
    root += "<data>"
    for i in range (len(train)):
        root += "<row> train/prediction: {}/{} </row>".format(train[i],prediction[i])

    root += "</data>"
    root += "</root>"
    #tree.write("Patients.xml", encoding="utf-8", xml_declaration=True, method="xml",pretty_print=True)
    dom = minidom.parseString(root)
    pretty_xml_str = dom.toprettyxml()
    now = datetime.now()
    dt_string = now.strftime("%Y.%m.%d_%H.%M")
    filename = "{}_{}.xml".format(filename, dt_string)
    with open(filename, "w") as f:
        f.write(pretty_xml_str)
def count_ones_and_zeros(array):
    unique_elements, counts = np.unique(array, return_counts=True)
    counts_dict = dict(zip(unique_elements, counts))

    count_ones = counts_dict.get(1, 0)
    count_zeros = counts_dict.get(0, 0)

    return count_zeros, count_ones

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

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

def data_cleaner(data: dict[str, pd.DataFrame], time_step: pd.Timedelta, missing_count_threshold,
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

if __name__ == "__main__":  # runs only if program was ran from this file, does not run when imported
    #data, patient_data = load(TEST2_540_PATH)
    #dropped_data = drop_days_with_missing_eat_data(data,3)
    #print('ok')
    #train,test = load_everything()
   #temp_data, temp_patient_data = load(TEST_559_PATH)
   #temp_data = fill_glucose_level_data_with_zeros(temp_data, pd.Timedelta(minutes=5))
   #temp_data = drop_meal_days(temp_data)
   #temp_data = fill_meal_with_zeros(temp_data)
    write_all_cleaned_xml()
    print(" ")
    #data_glucose = fill_glucose_level_data_with_zeros(data, pd.Timedelta(5,'m'))
    #data_filled = fill_meal_zeros(data_glucose,pd.Timedelta(5,'m'))
    print('.')
