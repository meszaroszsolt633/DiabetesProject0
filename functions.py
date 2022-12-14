import math
import ntpath
from xml_read import *
from xml_write import *
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



#nem biztos h sz??ks??g van r?? ||
#                            VV
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
    #kimentj??k a napokat
    current_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[0].date())
    last_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[-1].date())
    #v??gig megy??nk az ??sszes napon
    while current_day <= last_day:
        next_day = current_day + pd.Timedelta(1, 'd')
        #kimentj??k mindig az adott nap adatait a daybe
        day = cleaned_data['meal'][cleaned_data['meal']['ts'] >= current_day]
        day = day[day['ts'] < next_day]
        #ha a day ??res, vagy kevesebb adat van benne mint a threshold akkor kuka
        if day.empty or len(day) < missing_eat_threshold:
            for measurement_type in cleaned_data.keys():
                # if any timestamp is in the day that is to be thrown away, throw away the entire event
                for measurement_parameter in cleaned_data[measurement_type]:
                    if types[measurement_type][measurement_parameter] == 'datetime':
                        tdf = cleaned_data[measurement_type]
                        day_data = tdf[tdf[measurement_parameter] >= current_day]
                        day_data = day_data[measurement_parameter][day_data[measurement_parameter] < next_day]

                        cleaned_data[measurement_type] = cleaned_data[measurement_type].drop(index=day_data.index)
        #v??ltunk a k??vi napra
        current_day = next_day
    return cleaned_data

def insert_row(idx, df, df_insert):
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


def create_decreasing_rows_fixed(amount, datetime, avg):
    rows = pd.DataFrame(index=np.arange(0, amount), columns=('ts', 'value'))
    prev_datetime = datetime

    for i in range(0, amount):
        dt = pd.to_datetime(prev_datetime, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        rows.loc[i] = pd.Series({'ts': dt, 'value': math.floor(avg)})
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
    valueaftergap=int(valueaftergap)
    rows = pd.DataFrame(index=np.arange(0, amount), columns=('ts', 'value'))
    prev_datetime = datetime
    segment=(valueaftergap-avg)/amount
    value=avg

    for i in range(0, amount):
        dt = pd.to_datetime(prev_datetime, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        rows.loc[i] = pd.Series({'ts': dt, 'value': value})
        prev_datetime += pd.Timedelta(5, 'm')
        value=int(value+segment)
    return rows


def fill_start_glucose_level_data_continuous(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    avgs = avg_calculator(data)
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()

    # mindenekel??tt megn??zz??k hogy az els?? elem 0 ??ra k??rny??k??n van e
    first_timestamp = pd.Timestamp(year=cleaned_data['glucose_level']['ts'][0].year,
                                   month=cleaned_data['glucose_level']['ts'][0].month,
                                   day=cleaned_data['glucose_level']['ts'][0].day,
                                   hour=cleaned_data['glucose_level']['ts'][0].hour,
                                   minute=cleaned_data['glucose_level']['ts'][0].minute,
                                   second=cleaned_data['glucose_level']['ts'][0].second)
    # kimentj??k a 0 ??r??t egy v??ltoz??ba
    hour_zero = pd.Timestamp(year=cleaned_data['glucose_level']['ts'][0].year,
                             month=cleaned_data['glucose_level']['ts'][0].month,
                             day=cleaned_data['glucose_level']['ts'][0].day,
                             hour=0, minute=0, second=0)
    # megn??zz??k a k??l??nbs??get
    first_amount = first_timestamp - hour_zero
    if first_amount > pd.Timedelta(10, 'm'):
        # megn??zz??k mennyi elem hi??nyzik
        first_amount_missing = math.floor(first_amount.total_seconds() / time_step.total_seconds()) - 1
        df_to_insert = create_decreasing_rows_continuous(first_amount_missing, hour_zero, avgs[0],cleaned_data['glucose_level']['value'][0])
        cleaned_data['glucose_level'] = insert_row(0, cleaned_data['glucose_level'], df_to_insert)

    return cleaned_data
    ######################################


def fill_glucose_level_data_continuous(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    avgs = avg_calculator(data)
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    cleaned_data = fill_start_glucose_level_data_continuous(cleaned_data, time_step)
    avg_index = 0
    prev_ts = None
    #mivel a ciklusban nem friss??l az indexel??s, azaz a r??gi adatb??zison fut v??gig, kell 1 korrekci??
    #amivel a besz??r??st oldjuk meg. (ha besz??runk 5 elemet a 65 index ut??n, a 66. elem a r??gi 66. elem lesz
    # nem pedig az ??j besz??rt)
    corrector = 0
    for idx, ts in enumerate(cleaned_data['glucose_level']['ts']):
        if prev_ts is not None:
            dt = ts - prev_ts
            if ts.day != prev_ts.day:
                avg_index += 1
            if pd.Timedelta(1,'d') > dt >= time_step + time_step:
                # megn??zz??k mennyi hi??nyzik
                missing_amount = math.floor(dt.total_seconds() / time_step.total_seconds()) - 1

                # l??trehozunk 1 dataframe-t amiben megfelel?? mennyis??g?? sor van
                df_to_insert = create_increasing_rows_continuous(missing_amount, prev_ts,
                                                          cleaned_data['glucose_level']['value'][idx+corrector-1],cleaned_data['glucose_level']['value'][idx+corrector+1])
                # besz??rjuk az ??j dataframe??nket az eredetibe
                cleaned_data['glucose_level'] = insert_row(idx+corrector, cleaned_data['glucose_level'], df_to_insert)
                corrector += missing_amount
        prev_ts = ts
    return cleaned_data


def fill_start_glucose_level_data_fixed(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    avgs = avg_calculator(data)
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()

    # mindenekel??tt megn??zz??k hogy az els?? elem 0 ??ra k??rny??k??n van e
    first_timestamp = pd.Timestamp(year=data['glucose_level']['ts'][0].year,
                                   month=data['glucose_level']['ts'][0].month,
                                   day=data['glucose_level']['ts'][0].day,
                                   hour=data['glucose_level']['ts'][0].hour,
                                   minute=data['glucose_level']['ts'][0].minute,
                                   second=data['glucose_level']['ts'][0].second)
    # kimentj??k a 0 ??r??t egy v??ltoz??ba
    hour_zero = pd.Timestamp(year=data['glucose_level']['ts'][0].year,
                             month=data['glucose_level']['ts'][0].month,
                             day=data['glucose_level']['ts'][0].day,
                             hour=0, minute=0, second=0)
    # megn??zz??k a k??l??nbs??get
    first_amount = first_timestamp - hour_zero
    if first_amount > pd.Timedelta(10, 'm'):
        # megn??zz??k mennyi elem hi??nyzik
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
    #mivel a ciklusban nem friss??l az indexel??s, azaz a r??gi adatb??zison fut v??gig, kell 1 korrekci??
    #amivel a besz??r??st oldjuk meg. (ha besz??runk 5 elemet a 65 index ut??n, a 66. elem a r??gi 66. elem lesz
    # nem pedig az ??j besz??rt)
    corrector = 0
    for idx, ts in enumerate(data['glucose_level']['ts']):
        if prev_ts is not None:
            dt = ts - prev_ts
            if ts.day != prev_ts.day:
                avg_index += 1
            if pd.Timedelta(1,'d') > dt >= time_step + time_step:
                # megn??zz??k mennyi hi??nyzik
                missing_amount = math.floor(dt.total_seconds() / time_step.total_seconds()) - 1

                # l??trehozunk 1 dataframe-t amiben megfelel?? mennyis??g?? sor van
                df_to_insert = create_increasing_rows_fixed(missing_amount, prev_ts,
                                                          avgs[avg_index])
                # besz??rjuk az ??j dataframe??nket az eredetibe
                cleaned_data['glucose_level'] = insert_row(idx+corrector, cleaned_data['glucose_level'], df_to_insert)
                corrector += missing_amount
        prev_ts = ts
    return cleaned_data




def avg_calculator(data: dict[str, pd.DataFrame]):
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    glucose_level_average = []
    cleaned_data['glucose_level']['value'] = cleaned_data['glucose_level']['value'].astype(int)
    current_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[0].date())
    last_day = pd.to_datetime(cleaned_data['glucose_level']['ts'].iloc[-1].date())
    while current_day <= last_day:

        # Van 1 nem l??tez?? napunk 544-esben 2027-05-24 nap missing (: ez??rt NaN-t ad vissza...
        next_day = current_day + pd.Timedelta(1, 'd')

        glucose_level_count = cleaned_data['glucose_level'][cleaned_data['glucose_level']['ts'] >= current_day]
        glucose_level_count = glucose_level_count[glucose_level_count['ts'] < next_day]
        average = np.sum(glucose_level_count['value']) / glucose_level_count.shape[0]
        if not math.isnan(float(average)):
            glucose_level_average.append(math.floor(average))
        current_day = next_day
    return glucose_level_average

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


if __name__ == "__main__":  # runs only if program was ran from this file, does not run when imported
    data, patient_data = load(TRAIN2_544_PATH)
    filled_data = fill_glucose_level_data_continuous(data, pd.Timedelta(5,"m"))
    print(filled_data)

