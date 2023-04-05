import math
import ntpath
from xml_read import *
from xml_write import *
from defines import *
import pandas as pd
import numpy as np
import xml.dom.minidom as minidom
from model import *


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



#nem biztos h szükség van rá ||
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

def fill_glucose_level_data_with_zeros(data: dict[str, pd.DataFrame], time_step: pd.Timedelta):
    cleaned_data = {}
    for key in data.keys():
        cleaned_data[key] = data[key].__deepcopy__()
    prev_ts = None
    #mivel a ciklusban nem frissül az indexelés, azaz a régi adatbázison fut végig, kell 1 korrekció
    #amivel a beszúrást oldjuk meg. (ha beszúrunk 5 elemet a 65 index után, a 66. elem a régi 66. elem lesz
    # nem pedig az új beszúrt)
    corrector = 0
    for idx, ts in enumerate(cleaned_data['glucose_level']['ts']):
        if prev_ts is not None:
            dt = ts - prev_ts
            if pd.Timedelta(1,'d') > dt >= time_step + time_step:
                # megnézzük mennyi hiányzik
                missing_amount = math.floor(dt.total_seconds() / time_step.total_seconds()) - 1

                # létrehozunk 1 dataframe-t amiben megfelelő mennyiségű sor van
                df_to_insert = create_increasing_rows_fixed(missing_amount, prev_ts, 0)

                # beszúrjuk az új dataframeünket az eredetibe
                cleaned_data['glucose_level'] = insert_row(idx+corrector, cleaned_data['glucose_level'], df_to_insert)
                corrector += missing_amount
        prev_ts = ts
    return cleaned_data

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

if __name__ == "__main__":  # runs only if program was ran from this file, does not run when imported
    #data, patient_data = load(TEST2_567_PATH)
    #dropped_data = drop_days_with_missing_eat_data(data,3)
    #print('ok')
    count_glucose_level_data_overlapping(50)
    count_glucose_level_data_overlapping(60)
    count_glucose_level_data_overlapping(70)
    count_glucose_level_data_overlapping(80)


