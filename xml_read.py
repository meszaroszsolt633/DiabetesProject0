import pandas as pd
import xml.etree.ElementTree as ET

from defines import *
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


if __name__ == "__main__":  # runs only if program was ran from this file, does not run when imported
    data, patient_data = load(TRAIN2_540_PATH)
    print(data)

# 40.0
