

from xml_read import *
from functions import *
from defines import *


def write_to_xml(output_file_path: str, data: dict[str, pd.DataFrame], patient_id: int, insulin_type: str, body_weight: int = 99):
    root = ET.Element("patient")
    root.set('id', str(patient_id))
    root.set('weight', str(body_weight))
    root.set('insulin_type', insulin_type)

    for key in data.keys():
        new_element = ET.Element(key)

        for index, row in data[key].iterrows():
            new_event = ET.Element('event')

            for column in data[key]:
                if types[key][column] == 'datetime':
                    if pd.isnull(row[column]):
                        new_event.set(column, '')
                    else:
                        new_event.set(column, row[column].strftime(format='%d-%m-%Y %H:%M:%S'))
                else:
                    new_event.set(column, str(row[column]))

            new_element.append(new_event)

        root.append(new_element)

    tree = ET.ElementTree(root)

    with open(output_file_path, "wb") as file:
        tree.write(file, encoding='UTF-8', xml_declaration=True)


def filepath_to_string(filepath: str):
    return filepath[filepath.find('5'):filepath.rfind('.xml')] + 'Cleaned.xml'


if __name__ == "__main__":
    for file_path in ALL_FILE_PATHS:
        xml, patient = load(file_path)
        stringpath=filepath_to_string(file_path)
        cleaned = drop_days_with_missing_glucose_data(xml, 50)
        write_to_xml(os.path.join(CLEANED_DATA_DIR, stringpath), cleaned, int(patient['id']),patient['insulin_type'],body_weight=int(patient['weight']))



