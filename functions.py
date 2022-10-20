import math
import ntpath
from xml_read import *
from defines import *
import pandas as pd
import numpy as np


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


def get_file_missing_data_statistics(data: dict[str, pd.DataFrame]):
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

        if day.empty or get_file_missing_data_statistics({'glucose_level': day})['glucose_level'][
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



def insert_row(idx, df, df_insert):
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]

    df = pd.concat([dfA,df_insert,dfB])

    return df


def create_increasing_rows(amount, datetime, avg, idx):
    rows = pd.DataFrame(index=np.arange(idx+1, idx+amount), columns=('ts', 'value'))
    next_datetime = datetime

    for i in range(0, amount):
        next_datetime += pd.Timedelta(5, 'm')
        dt = pd.to_datetime(next_datetime, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        rows.loc[i] = pd.Series({'ts':dt, 'value' : avg})
    return rows


def fill_glucose_level_data(data: dict[str, pd.DataFrame], time_step: pd.Timedelta,):
    avgs = avg_calculator(data)
    datacopy = data
    current_day = pd.to_datetime(datacopy['glucose_level']['ts'].iloc[0].date())
    last_day = pd.to_datetime(datacopy['glucose_level']['ts'].iloc[-1].date())
    prev_ts = None
    for idx, ts in enumerate(datacopy['glucose_level']['ts']):
        if prev_ts is not None:
            dt = ts - prev_ts
            if dt >= time_step + pd.Timedelta(5, 'm'):
                #megnézzük mennyi hiányzik
                missing_amount = math.floor(dt.total_seconds() / time_step.total_seconds()) - 1
                #létrehozunk 1 dataframe-t amiben megfelelő mennyiségű sor van
                df_to_insert = create_increasing_rows(missing_amount, prev_ts, avgs[0], idx) #Average-t még meg kell oldani, hogy mi
                #beszúrjuk az új dataframeünket az eredetibe
                datacopy_glucose = insert_row(idx,datacopy['glucose_level'], df_to_insert)
                datacopy['glucose_level'] = datacopy_glucose
        prev_ts = ts
    return datacopy


def avg_calculator(data: dict[str, pd.DataFrame]):
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    glucose_level_average = []
    cleaned_data['glucose_level']['value']= cleaned_data['glucose_level']['value'].astype(int)
    current_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[0].date())
    last_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[-1].date())
    while current_day <= last_day:
        next_day = current_day + pd.Timedelta(1, 'd')

        glucose_level_count=cleaned_data['glucose_level'][cleaned_data['glucose_level']['ts']>=current_day]
        glucose_level_count=glucose_level_count[glucose_level_count['ts']< next_day]
        average=np.sum(glucose_level_count['value'])/glucose_level_count.shape[0]
        glucose_level_average.append(average)
        current_day = next_day
    return glucose_level_average


def last_existing_value(data, time_step: pd.Timedelta):
    prev_ts = None
    prevTimeStep=False
    ts_dict= {'last_existing_before_hole': [] ,"first_existing_after_hole": [] }
    for ts in data['glucose_level']['ts']:
        if prev_ts is not None:
            dt = ts - prev_ts
            if dt == time_step or dt < time_step:
                prevTimeStep=True
            elif dt > (time_step * 1.5) and prevTimeStep==True:
                ts_dict['last_existing_before_hole'].append(prev_ts)
                ts_dict['first_existing_after_hole'].append(ts)
        prev_ts=ts
    return ts_dict




if __name__ == "__main__":  # runs only if program was ran from this file, does not run when imported
    data, patient_data = load(TRAIN2_544_PATH)
    print(data['glucose_level'])

