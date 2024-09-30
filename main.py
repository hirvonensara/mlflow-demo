# EXTERNAL IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
import sys
import os
import yaml

# INTERNAL IMPORTS
from utils import data_preprocessing

fd = Path(__file__).parent
sys.path.append(os.path.abspath(os.path.dirname(fd)))
fd = Path(__file__).parent.parent
sys.path.append(os.path.abspath(os.path.dirname(fd)))


def create_raw_dataset(paths, result_path):
    df = data_preprocessing.merge_fmi_data(paths, result_path=result_path)
    return df


def main():
    src_path = Path(__file__).parent
    path = os.path.abspath(src_path)

    # CREATE RAW DATA SET
    # FIRST, WEATHER
    paths = [
        path + "\data\\raw\kumpula_weather_1.1.2021_30.9.2021.csv",
        path + '\data\\raw\kumpula_weather_1.10.2021_1.10.2022.csv',
        path + '\data\\raw\kumpula_weather_2.10.2022_30.9.2023.csv'

    ]
    # check if exists
    raw_weather_path = path + '/data/raw/kumpula_weather.csv'

    if np.logical_not(os.path.exists(raw_weather_path)):
        print("File doesn't exist, created new one.")
        df_weather = create_raw_dataset(paths, raw_weather_path)

    else:
        df_weather = pd.read_csv(raw_weather_path)

    print(df_weather.head())



    






main()