import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#from statsmodels.tsa.seasonal import STL, MSTL
import matplotlib.dates as mdates
from pathlib import Path 
import sys
import os
import yaml

fd = Path(__file__).parent
sys.path.append(os.path.abspath(os.path.dirname(fd)))
fd = Path(__file__).parent.parent
sys.path.append(os.path.abspath(os.path.dirname(fd)))

def merge_fmi_data(source_paths, result_path):
    dfs = []
    for path in source_paths:
        df = pd.read_csv(path)
        df = df.drop(['Observation station'], axis=1)
        df[['hour', 'minute']] = df['Time [UTC]'].str.split(':', expand=True).astype(int)
        df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'hour', 'minute']])
        # set date column index
        df = df.set_index('date')
        dfs.append(df)

    # stack dataframes
    result = pd.concat(dfs)
    result.to_csv(result_path)

    return result


def visualise(df, y_col):
    #sns.set_style('whitegrid')
    plt.figure(figsize=(6,3))
    plt.plot(df.index, df[y_col], linewidth=0.8)
    ax = plt.gca()
    locator = mdates.AutoDateLocator(tz='UTC', minticks=8, maxticks=15)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    plt.xticks(rotation=45)
    ax.set_xlabel('time')
    ax.set_ylabel(y_col)
    ax.set_title(y_col)
    plt.show()

def deal_with_nulls(df, remove_all=False, impute_with_prev_n_days=1, impute_with_prev_n_hours=None):
    print('Nulls before removing:')
    print(df.isna().sum())

    # jos ei yhtään nullia? --> no dropna ja fillna ei sit vaan tee mitää
    while df.isna().sum().sum() != 0:
        # poistetaan kaikki
        if remove_all:
            print('Removed all nans')
            df = df.dropna()

        elif impute_with_prev_n_days != None:
            print('Filled nans with values from', impute_with_prev_n_days, 'days')
            # impute nans with previous values
            df = df.fillna(df.shift(impute_with_prev_n_days, freq='D'))

        elif impute_with_prev_n_hours != None:
            print('Filled nans with values from', impute_with_prev_n_hours, 'hours')
            df = df.fillna(df.shift(impute_with_prev_n_hours, freq='H'))

    return df


def remove_outlier_iq_range(df, target, visualise=True, fill_with_previous_n_days = 1):
    Q3 = np.quantile(df[target].dropna(), 0.75)
    Q1 = np.quantile(df[target].dropna(), 0.25)
    IQR = Q3 - Q1
    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR
    print('Lower range', lower_range)
    print('Upper range', upper_range)

    outliers_idx = df[(df[target]>upper_range) | (df[target]<lower_range)].index

    print('Number of outliers found', len(outliers_idx))

    if visualise:
        mark_idx = df.index.get_indexer(outliers_idx)
        #sns.set_style('whitegrid')
        fig, ax = plt.subplots(1, figsize=(8,5))

        ax.plot(df.index, df[target], '-gD', markevery=mark_idx, mfc='white', mec='red')
        ax.xaxis.set_tick_params(rotation=45)
        plt.show()

    df.loc[outliers_idx, [target]] = np.nan

    if fill_with_previous_n_days != None:
        df = deal_with_nulls(df, impute_with_prev_n_days=fill_with_previous_n_days)

    else:
        # drop outliers
        df = df.dropna()


    return df

def filter_outliers_by_rules(df, rules, target, visualise=True, fill_with_previous_n_days=None):
    outlier_list = []

    for rule in rules.values():
        idx = df.loc[rule['start']:rule['end']].loc[df[target]>rule['threshold']].index


        outlier_list.append(idx)

    outlier_idx = pd.DatetimeIndex(pd.concat([pd.Series(idx) for idx in outlier_list])).sort_values()

    print('Number of outliers found', len(outlier_idx))

    if visualise:
        mark_idx = df.index.get_indexer(outlier_idx)
        print(len(mark_idx))
        #sns.set_style('whitegrid')
        fig, ax = plt.subplots(1, figsize=(8,5))

        ax.plot(df.index, df[target], '-gD', markevery=mark_idx, mfc='white', mec='red')
        ax.xaxis.set_tick_params(rotation=45)
        plt.show()

    df.loc[outlier_idx, target] = np.nan

    if fill_with_previous_n_days != None:
        df = deal_with_nulls(df, impute_with_prev_n_days=fill_with_previous_n_days)

    else:
        # drop outliers
        df = df.dropna()

    return df

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

