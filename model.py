import pandas as pd

from defines import *
from functions import *
from xml_read import *
from xml_write import *
import tensorflow as tf
from tensorflow import keras

def data_preparation(data: dict[str, pd.DataFrame], time_step: pd.Timedelta, missing_count_threshold, missing_eat_threshold)-> dict[str, pd.DataFrame]:
    cleaned_data = drop_days_with_missing_glucose_data(data, missing_count_threshold)
    cleaned_data = drop_days_with_missing_eat_data(cleaned_data, missing_eat_threshold)
    cleaned_data = fill_glucose_level_data_continuous(cleaned_data, time_step)
    return cleaned_data

#if __name__ == "__main__":
