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
