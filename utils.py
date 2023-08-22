import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_data(path_to_files: str, columns: list):
    """
    This script only load data files scattered by waves in the folders.
    """
    all_files = glob.glob(os.path.join(path_to_files,
                                       '*indresp.dta'))
    indresp = []
    prefixes = [] # for later
    for filename in all_files:
        prefix = filename.split('/')[-1][0:2]
        prefixes.append(prefix)
        colnames = [f'{prefix}{x}' for x in columns]
        temp_df = pd.read_stata(filename,
                       columns=['pidp'] + colnames)
        indresp.append(temp_df)
    for i, df in enumerate(indresp):
        df['wave'] = i+1
        df.columns = ['pidp'] + columns + ['wave'] # the order of the columns here is importan
    out = pd.concat(indresp)
    out['max_waves'] = out.groupby('pidp')['wave'].transform('count').values
    return out


def create_index(x):
    """
    Create relative index based on length of array.
    """
    y = np.arange(len(x)) + 1
    return y - x
642

def recoding_and_cleaning(in_data, cpih_data):
    """
    This function recodes and clean various variables
    Is in general a mess that runs once.
    """
    data = in_data.copy()
    data['month_recoded'] = data.month.str.strip().replace(
        {"jan yr1": 1,
         "feb yr1": 2,
         "mar yr1": 3,
         "apr yr1": 4,
         "may yr1": 5,
         "jun yr1": 6,
         "jul yr1": 7,
         "aug yr1": 8,
         "sep yr1": 9,
         "oct yr1": 10,
         "nov yr1": 11,
         "dec yr1": 12,
         "jan yr2": 13,
         "feb yr2": 14,
         "mar yr2": 15,
         "apr yr2": 16,
         "may yr2": 17,
         "jun yr2": 18,
         "jul yr2": 19,
         "aug yr2": 20,
         "sep yr2": 21,
         "oct yr2": 22,
         "nov yr2": 23,
         "dec yr2": 24,
         "Not available for IEMB": np.nan})
    data['month_for_date'] = data.month.str.strip().replace(
        {"jan yr1": 1,
         "feb yr1": 2,
         "mar yr1": 3,
         "apr yr1": 4,
         "may yr1": 5,
         "jun yr1": 6,
         "jul yr1": 7,
         "aug yr1": 8,
         "sep yr1": 9,
         "oct yr1": 10,
         "nov yr1": 11,
         "dec yr1": 12,
         "jan yr2": 1,
         "feb yr2": 2,
         "mar yr2": 3,
         "apr yr2": 4,
         "may yr2": 5,
         "jun yr2": 6,
         "jul yr2": 7,
         "aug yr2": 8,
         "sep yr2": 9,
         "oct yr2": 10,
         "nov yr2": 11,
         "dec yr2": 12,
         "Not available for IEMB": np.nan})
    data['year'] = np.repeat(np.nan, data.shape[0])
    data.loc[(data.month_recoded<=12)&(data.wave==1), 'year'] = int(2009)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==1), 'year'] = int(2010)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==2), 'year'] = int(2010)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==2), 'year'] = int(2011)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==3), 'year'] = int(2011)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==3), 'year'] = int(2012)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==4), 'year'] = int(2012)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==4), 'year'] = int(2013)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==5), 'year'] = int(2013)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==5), 'year'] = int(2014)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==6), 'year'] = int(2014)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==6), 'year'] = int(2015)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==7), 'year'] = int(2015)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==7), 'year'] = int(2016)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==8), 'year'] = int(2016)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==8), 'year'] = int(2017)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==9), 'year'] = int(2017)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==9), 'year'] = int(2018)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==10), 'year'] = int(2018)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==10), 'year'] = int(2019)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==11), 'year'] = int(2019)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==11), 'year'] = int(2020)
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==12), 'year'] = int(2020)
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==12), 'year'] = int(2021)
    data['date'] = data['year'].astype(str).str.split('.').str[0] + '-' + data['month_for_date'].astype(str).str.split('.').str[0] + '-' + '1'
    for index, row in data.iterrows():
        try:
            row['date'] = pd.to_datetime(row['date'], format='%Y-%m-%d')
        except ValueError:
            row['date'] = np.nan
    data['aidhh_recoded'] = data.aidhh.replace({'Yes': 'yes', # this variable encodes caring at home
                                                'No': 'no',
                                                'no     ': 'no',
                                                'yes    ': 'yes',
                                                'Not available for IEMB': np.nan,
                                                "don't know": np.nan,
                                                'proxy': np.nan,
                                                'missing': np.nan,
                                                'refusal': np.nan,
                                                'inapplicable': np.nan})
    data['aidxhh_recoded'] = data.aidxhh.replace({'Yes': 'yes', # this variable encodes caring outisde home
                                                'No': 'no',
                                                'no     ': 'no',
                                                'yes    ': 'yes',
                                                'Not available for IEMB': np.nan,
                                                "don't know": np.nan,
                                                'proxy': np.nan,
                                                'missing': np.nan,
                                                'refusal': np.nan,
                                                'inapplicable': np.nan})
    data['aidhrs_recoded_3'] = data.aidhrs.str.strip().replace({"inapplicable": np.nan, # this variable encodes hours caring in 3 categories (this might change in the future)
                                   "0 - 4 hours per week": "0-19",
                                   "proxy": np.nan,
                                   "5 - 9 hours per week": "0-19",
                                   "10 - 19 hours per week": "0-19",
                                   "100 or more hours per week/continuous care": "50+",
                                   "20 - 34 hours per week": "20-49",
                                   "35 - 49 hours per week": "20-49",
                                   "varies 20 hours or more": "20-49",
                                   "0 - 4 hours per week": "0-19",
                                   "10-19 hours per week": "0-19",
                                   "Not available for IEMB": np.nan,
                                   "Varies 20 hours or more": "20-49",
                                   "varies under 20 hours": "0-19",
                                   "Varies under 20 hours": "0-19",
                                   "20-34 hours per week": "20-49",
                                   "5 - 9 hours per week": "0-19",
                                   "Other": np.nan,
                                   "10 - 19 hours per week": "0-19",
                                   "35-49 hours per week": "20-49",
                                   "100 or more hours per week/continuous care": "50+",
                                   "20 - 34 hours per week": "20-49",
                                   "50-99 hours per week": "50+",
                                   "other": np.nan,
                                   "don't know": np.nan,
                                   "35 - 49 hours per week": "20-49",
                                   "varies under 20 hours": "0-19",
                                   "Varies 20 hours or more": "20-49",
                                   "50 - 99 hours per week": "50+",
                                   "other": np.nan,
                                   "refusal": np.nan})
    data['jbstat_clean'] = data.jbstat.str.strip().str.lower()
    data['employed'] = data.jbstat_clean.replace({"paid employment(ft/pt)": "employed", # This variable encodes employed/unemployed status
                                                  "retired": "unemployed",
                       "self employed": "employed",
                       "family care or home": "unemployed",
                       "unemployed": "unemployed",
                       "full-time student": "unemployed",
                       "lt sick or disabled": "unemployed",
                       "on maternity leave": "employed",
                       "doing something else": "unemployed",
                       "on furlough": "unemployed",
                       "on apprenticeship": "employed",
                       "unpaid, family business": "unemployed",
                       "govt training scheme": "employed",
                       "refusal": np.nan,
                       "temporarily laid off/short term working": "unemployed",
                       "don't know": np.nan,
                       "missing": np.nan}
                     )
    data['wage'] = data.paygu_dv.replace({'inapplicable': np.nan, 'proxy': np.nan, 'missing': np.nan})
    cpih_data['date'] = pd.to_datetime(cpih_data.date, format='%b-%y')
    data['istrtdaty'] = data['istrtdaty'].astype('str').replace({'inapplicable': np.nan, 'missing': np.nan, "don't know": np.nan})
    data['istrtdatm'] = data['istrtdatm'].replace({'inapplicable': np.nan, 'missing': np.nan, "don't know": np.nan})
    data['date'] = data['istrtdaty'].astype(str) + '/' + data['istrtdatm'].astype(str)
    data['date'] = data['date'].replace({'nan/nan': np.nan})
    data['date'] = pd.to_datetime(data['date'], format='%Y/%B')
    data = data.merge(cpih_data, on='date', how='left')
    data['jbhrs_clean'] = data.jbhrs.replace({'inapplicable': np.nan, 'proxy': np.nan, 'missing': np.nan, "don't know": np.nan, "refusal": np.nan, 0: np.nan})
    data['jbhrs_clean'][data.jbhrs_clean < 1] = np.nan
    data['month_jbhrs'] = data['jbhrs_clean'] * 4.33 # times the average amount of weeks
    data['wage_h'] = data['wage'] / data['month_jbhrs']
    data['wage_h'] = data['wage'] / data['month_jbhrs']
    data['wage_h_deflated'] = (data['wage_h'] / data['cpih']) * 100
    data['log_wage_h_deflated'] = np.log(data['wage_h_deflated'])
    return data
