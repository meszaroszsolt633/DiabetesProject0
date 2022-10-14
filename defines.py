import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(ROOT_DIR, 'Test')
TEST2_DIR = os.path.join(ROOT_DIR, 'Test2')
TRAIN_DIR = os.path.join(ROOT_DIR, 'Train')
TRAIN2_DIR = os.path.join(ROOT_DIR, 'Train2')
CLEANED_DATA_DIR=os.path.join(ROOT_DIR,'Cleaned Data')



TRAIN_559_PATH = os.path.join(TRAIN_DIR, '559-ws-training.xml')
TRAIN_563_PATH = os.path.join(TRAIN_DIR, '563-ws-training.xml')
TRAIN_570_PATH = os.path.join(TRAIN_DIR, '570-ws-training.xml')
TRAIN_575_PATH = os.path.join(TRAIN_DIR, '575-ws-training.xml')
TRAIN_588_PATH = os.path.join(TRAIN_DIR, '588-ws-training.xml')
TRAIN_591_PATH = os.path.join(TRAIN_DIR, '591-ws-training.xml')

TRAIN2_540_PATH = os.path.join(TRAIN2_DIR, '540-ws-training.xml')
TRAIN2_544_PATH = os.path.join(TRAIN2_DIR, '544-ws-training.xml')
TRAIN2_552_PATH = os.path.join(TRAIN2_DIR, '552-ws-training.xml')
TRAIN2_567_PATH = os.path.join(TRAIN2_DIR, '567-ws-training.xml')
TRAIN2_596_PATH = os.path.join(TRAIN2_DIR, '596-ws-training.xml')

TRAIN_FILE_PATHS = [TRAIN_559_PATH, TRAIN_563_PATH, TRAIN_570_PATH, TRAIN_575_PATH, TRAIN_588_PATH, TRAIN_591_PATH]
TRAIN2_FILE_PATHS = [TRAIN2_540_PATH, TRAIN2_544_PATH, TRAIN2_552_PATH, TRAIN2_567_PATH, TRAIN2_596_PATH]
ALL_TRAIN_FILE_PATHS = TRAIN_FILE_PATHS + TRAIN2_FILE_PATHS


TEST_559_PATH = os.path.join(TEST_DIR, '559-ws-testing.xml')
TEST_563_PATH = os.path.join(TEST_DIR, '563-ws-testing.xml')
TEST_570_PATH = os.path.join(TEST_DIR, '570-ws-testing.xml')
TEST_575_PATH = os.path.join(TEST_DIR, '575-ws-testing.xml')
TEST_588_PATH = os.path.join(TEST_DIR, '588-ws-testing.xml')
TEST_591_PATH = os.path.join(TEST_DIR, '591-ws-testing.xml')

TEST2_540_PATH = os.path.join(TEST2_DIR, '540-ws-testing.xml')
TEST2_544_PATH = os.path.join(TEST2_DIR, '544-ws-testing.xml')
TEST2_552_PATH = os.path.join(TEST2_DIR, '552-ws-testing.xml')
TEST2_567_PATH = os.path.join(TEST2_DIR, '567-ws-testing.xml')
TEST2_584_PATH = os.path.join(TEST2_DIR, '584-ws-testing.xml')
TEST2_596_PATH = os.path.join(TEST2_DIR, '596-ws-testing.xml')

TEST_FILE_PATHS = [TEST_559_PATH, TEST_563_PATH, TEST_570_PATH, TEST_575_PATH, TEST_588_PATH, TEST_591_PATH]
TEST2_FILE_PATHS = [TEST2_540_PATH, TEST2_544_PATH, TEST2_552_PATH, TEST2_567_PATH, TEST2_584_PATH, TEST2_596_PATH]
ALL_TEST_FILE_PATHS = TEST_FILE_PATHS + TEST2_FILE_PATHS

ALL_FILE_PATHS = ALL_TRAIN_FILE_PATHS + ALL_TEST_FILE_PATHS


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
