import pandas as pd
import xml.etree.ElementTree as ET

from root_defines import *
from typing import Union


# ID
# Weight: 99 is a placeholder, the actual data is unavailable
# glucose_level: continouse monitoring data every 5 minutes
# finger_stick: glucose values obtained by the patient
# basal: background insulin
# temp_basal: temporal basal insulin; 0= suspended
# bolus: insulin before meals, two types:
#  normal(all insulin at once)
#  stretched: over time
# meal: type of the meal and the carbohydrate estimate
# sleep: times of the reported sleep and the sleep quality:
#  1 for Poor; 2 for Fair; 3 for good
# work: Times of work and the intensity(1-10)
# stressors: Time of reported stress
# hypo_event: time of hypoglycemic episode
# illness: time of illness
# exercise: time and intensity(1-10) of exercise
# basis_heart_rate: heart rate every 5 minutes
# basis_gsr: galvanic skin response every 5 or 1 minutes
# basis_skin_temperature: skin temperature in fahrenheit every 5 or 1 minutes
# basis_air_temperature: air temperature every 5 or 1 minutes
# basis_steps: step count every 5 minutes
# basis_sleep: times reported that the subject was asleep
# acceleration: magnitude of acceleration


def load_xml(file_path: str) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:

    xml_doc = ET.parse(file_path).getroot()

    converted_data: dict[str, dict[str, list[str]]] = {}

    for measurement_type in xml_doc:
        # initialize dict
        converted_data[measurement_type.tag] = {}

        for measurement in measurement_type:
            for attribute in measurement.attrib:
                # initialize list if necessary
                if converted_data[measurement_type.tag].get(attribute) is None:
                    converted_data[measurement_type.tag][attribute] = []

                converted_data[measurement_type.tag][attribute].append(measurement.attrib[attribute])

    return converted_data, xml_doc.attrib


ts_tags = [
    'ts',
    'ts_begin',
    'ts_end',
    'tbegin',
    'tend'
]

string_tags = [
    'type',
    'competitive',
    'description',
    'name'
]

# lists the data types of all data in the xml files
types: dict[str, dict[str, str]] = {
    # <event ts="17-01-2022 00:04:00" value="135"/>
    'glucose_level': {'ts': 'datetime', 'value': 'int'},

    # <event ts="16-01-2022 20:11:38" value="169"/>
    'finger_stick': {'ts': 'datetime', 'value': 'int'},

    # <event ts="16-01-2022 17:00:00" value="0.88"/>
    'basal': {'ts': 'datetime', 'value': 'float'},

    # <event ts_begin="31-12-2021 00:32:21" ts_end="31-12-2021 02:32:00" value="0.0"/>
    'temp_basal': {'ts_begin': 'datetime', 'ts_end': 'datetime', 'value': 'float'},

    # <event ts_begin="07-12-2021 07:36:54" ts_end="07-12-2021 07:36:54" type="normal dual" dose="8.0" bwz_carb_input="102"/>
    'bolus': {'ts_begin': 'datetime', 'ts_end': 'datetime', 'type': 'str', 'dose': 'float', 'bwz_carb_input': 'int'},

    # <event ts="07-12-2021 18:28:00" type="Dinner" carbs="65"/>
    'meal': {'ts': 'datetime', 'type': 'str', 'carbs': 'int'},

    # <event ts_begin="08-12-2021 04:50:00" ts_end="07-12-2021 22:09:00" quality="3"/>
    'sleep': {'ts_begin': 'datetime', 'ts_end': 'datetime', 'quality': 'int'},

    # <event ts_begin="08-12-2021 05:20:00" ts_end="08-12-2021 15:48:00" intensity="1"/>
    'work': {'ts_begin': 'datetime', 'ts_end': 'datetime', 'intensity': 'int'},

    # <event ts="22-12-2021 17:03:00" type=" " description=" "/>
    'stressors': {'ts': 'datetime', 'type': 'str', 'description': 'str'},

    # <event ts="11-12-2021 21:45:00">  !!!Has "symptom" subtag
    'hypo_event': {'ts': 'datetime'},

    # <event ts_begin="12-12-2021 10:12:00" ts_end="" type="" description=" "/>
    'illness': {'ts_begin': 'datetime', 'ts_end': 'datetime', 'type': 'str', 'description': 'str'},

    # <event ts="13-12-2021 13:55:00" intensity="3" type=" " duration="150" competitive=""/>
    'exercise': {'ts': 'datetime', 'intensity': 'int', 'type': 'str', 'duration': 'int', 'competitive': 'str'},

    # <event ts="07-12-2021 14:51:00" value="117"/>
    'basis_heart_rate': {'ts': 'datetime', 'value': 'int'},

    # <event ts="07-12-2021 14:55:00" value="6.8E-5"/>
    'basis_gsr': {'ts': 'datetime', 'value': 'float'},

    # <event ts="07-12-2021 14:55:00" value="86.54"/>
    'basis_skin_temperature': {'ts': 'datetime', 'value': 'float'},

    # <event ts="07-12-2021 14:55:00" value="83.12"/>
    'basis_air_temperature': {'ts': 'datetime', 'value': 'float'},

    # <event ts="07-12-2021 14:51:00" value="0"/>
    'basis_steps': {'ts': 'datetime', 'value': 'int'},

    # <event tbegin="07-12-2021 22:57:00" tend="07-12-2021 22:59:00" quality="89" type=" "/>
    'basis_sleep': {'tbegin': 'datetime', 'tend': 'datetime', 'quality': 'int', 'type': 'str'},

    # <event ts="19-05-2027 09:55:00" value="0.9789230227470398"/>
    'acceleration': {'ts': 'datetime', 'value': 'float'},
}


def convert_to_pandas_dataframe(data: dict[str, dict[str, list[str]]]) -> dict[str, pd.DataFrame]:
    converted_data: dict[str, pd.DataFrame] = {}

    for measurement_type in data:
        converted_data[measurement_type] = pd.DataFrame.from_dict(data[measurement_type])

        for measurement_data in converted_data[measurement_type].columns:
            # convert data to the appropriate type
            if types[measurement_type][measurement_data] == 'datetime':
                converted_data[measurement_type][measurement_data] = \
                    pd.to_datetime(converted_data[measurement_type][measurement_data],
                                   format='%d-%m-%Y %H:%M:%S',
                                   errors='coerce')
            elif types[measurement_type][measurement_data] == 'int':
                converted_data[measurement_type][measurement_data].astype(int)
            elif types[measurement_type][measurement_data] == 'float':
                converted_data[measurement_type][measurement_data].astype(float)
            elif types[measurement_type][measurement_data] == 'str':
                pass
            else:
                raise NotImplementedError(f'type not found for {measurement_type} => {measurement_data}')

    return converted_data


def load(file_path: str) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    data, patient_data = load_xml(file_path)
    return convert_to_pandas_dataframe(data), patient_data


def transform_xml(xmldoc: Union[ET.Element, ET.ElementTree]):
    patientattr = {}
    for event in xmldoc.find('glucose_level'):
        t_dict = patientattr.copy()
        t_dict['glucose_level_ts'] = event.attrib['ts']
        t_dict['glucose_level_value'] = event.attrib['value']
        yield t_dict
    for event in xmldoc.find('finger_stick'):
        t_dict = patientattr.copy()
        t_dict['finger_stick_ts'] = event.attrib['ts']
        t_dict['finger_stick_value'] = event.attrib['value']
        yield t_dict
    for event in xmldoc.find('basal'):
        t_dict = patientattr.copy()
        t_dict['basal_ts'] = event.attrib['ts']
        t_dict['basal_value'] = event.attrib['value']
        yield t_dict
    for event in xmldoc.find('temp_basal'):
        t_dict = patientattr.copy()
        t_dict['temp_basal_ts_begin'] = event.attrib['ts_begin']
        t_dict['temp_basal_ts_end'] = event.attrib['ts_end']
        t_dict['temp_basal_value'] = event.attrib['value']
        yield t_dict
    for event in xmldoc.find('bolus'):
        t_dict = patientattr.copy()
        t_dict['bolus_ts_begin'] = event.attrib['ts_begin']
        t_dict['bolus_ts_end'] = event.attrib['ts_end']
        t_dict['bolus_type'] = event.attrib['type']
        t_dict['bolus_dose'] = event.attrib['dose']
        yield t_dict
    for event in xmldoc.find('meal'):
        t_dict = patientattr.copy()
        t_dict['meal_ts'] = event.attrib['ts']
        t_dict['meal_type'] = event.attrib['type']
        t_dict['meal_carbs'] = event.attrib['carbs']
        yield t_dict
    for event in xmldoc.find('sleep'):
        t_dict = patientattr.copy()
        t_dict['sleep'] = event.attrib
        yield t_dict
    for event in xmldoc.find('work'):
        t_dict = patientattr.copy()
        t_dict['work'] = event.attrib
        yield t_dict
    for event in xmldoc.find('stressors'):
        t_dict = patientattr.copy()
        t_dict['stressors'] = event.attrib
        yield t_dict
    for event in xmldoc.find('hypo_event'):
        t_dict = patientattr.copy()
        t_dict['hypo_event'] = event.attrib
        yield t_dict
    for event in xmldoc.find('illness'):
        t_dict = patientattr.copy()
        t_dict['illness'] = event.attrib
        yield t_dict
    for event in xmldoc.find('exercise'):
        t_dict = patientattr.copy()
        t_dict['exercise'] = event.attrib
        yield t_dict
    for event in xmldoc.find('basis_heart_rate'):
        t_dict = patientattr.copy()
        t_dict['basis_heart_rate'] = event.attrib
        yield t_dict
    for event in xmldoc.find('basis_gsr'):
        t_dict = patientattr.copy()
        t_dict['basis_gsr_ts'] = event.attrib['ts']
        t_dict['basis_gsr_value'] = event.attrib['value']
        yield t_dict
    for event in xmldoc.find('basis_skin_temperature'):
        t_dict = patientattr.copy()
        t_dict['basis_skin_temperature_ts'] = event.attrib['ts']
        t_dict['basis_skin_temperature_value'] = event.attrib['value']
        yield t_dict
    for event in xmldoc.find('basis_air_temperature'):
        t_dict = patientattr.copy()
        t_dict['basis_air_temperature'] = event.attrib
        yield t_dict
    for event in xmldoc.find('basis_sleep'):
        t_dict = patientattr.copy()
        t_dict['basis_sleep_ts_begin'] = event.attrib['tbegin']
        t_dict['basis_sleep_ts_end'] = event.attrib['tend']
        yield t_dict
    for event in xmldoc.find('acceleration'):
        t_dict = patientattr.copy()
        t_dict['acceleration_ts'] = event.attrib['ts']
        t_dict['acceleration_value'] = event.attrib['value']
        yield t_dict


def load_file(file_path: str) -> pd.DataFrame:
    tree = ET.parse(file_path)
    root = tree.getroot()
    trans = transform_xml(root)
    df = pd.DataFrame(list(trans))
    df['glucose_level_ts'] = pd.to_datetime(df['glucose_level_ts'], format='%d-%m-%Y %H:%M:%S')
    df['finger_stick_ts'] = pd.to_datetime(df['finger_stick_ts'], format='%d-%m-%Y %H:%M:%S')
    df['basal_ts'] = pd.to_datetime(df['basal_ts'], format='%d-%m-%Y %H:%M:%S')
    df['temp_basal_ts_begin'] = pd.to_datetime(df['temp_basal_ts_begin'], format='%d-%m-%Y %H:%M:%S')
    df['temp_basal_ts_end'] = pd.to_datetime(df['temp_basal_ts_end'], format='%d-%m-%Y %H:%M:%S')
    df['bolus_ts_begin'] = pd.to_datetime(df['bolus_ts_begin'], format='%d-%m-%Y %H:%M:%S')
    df['bolus_ts_end'] = pd.to_datetime(df['bolus_ts_end'], format='%d-%m-%Y %H:%M:%S')
    df['basis_gsr_ts'] = pd.to_datetime(df['basis_gsr_ts'], format='%d-%m-%Y %H:%M:%S')
    df['basis_skin_temperature_ts'] = pd.to_datetime(df['basis_skin_temperature_ts'], format='%d-%m-%Y %H:%M:%S')
    df['basis_sleep_ts_begin'] = pd.to_datetime(df['basis_sleep_ts_begin'], format='%d-%m-%Y %H:%M:%S')
    df['basis_sleep_ts_end'] = pd.to_datetime(df['basis_sleep_ts_end'], format='%d-%m-%Y %H:%M:%S')
    df['acceleration_ts'] = pd.to_datetime(df['acceleration_ts'], format='%d-%m-%Y %H:%M:%S')
    df['meal_ts'] = pd.to_datetime(df['meal_ts'], format='%d-%m-%Y %H:%M:%S')
    df['glucose_level_value'] = df['glucose_level_value'].astype(float)
    df['finger_stick_value'] = df['finger_stick_value'].astype(float)
    df['basal_value'] = df['basal_value'].astype(float)
    df['temp_basal_value'] = df['temp_basal_value'].astype(float)
    df['meal_carbs'] = df['meal_carbs'].astype(float)
    df['basis_gsr_value'] = df['basis_gsr_value'].astype(float)
    df['bolus_dose'] = df['bolus_dose'].astype(float)
    df['basis_skin_temperature_value'] = df['basis_skin_temperature_value'].astype(float)
    df['acceleration_value'] = df['acceleration_value'].astype(float)

    return df


if __name__ == "__main__":  # runs only if program was ran from this file, does not run when imported
    file = load_file(TRAIN2_540_PATH)

    maximum = 200
    for x in file['glucose_level_value']:
        if x < maximum:
            maximum = x
    print(maximum)


# 40.0
