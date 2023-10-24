import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
from operator import add
from toolz import reduce, partial
from scipy.optimize import minimize
from sklearn.neighbors import KDTree
from multiprocessing import Pool


def load_data(path_to_files: str, columns: list):
    """
    This script only load data files scattered by waves in the folders.
    """
    all_files = glob.glob(os.path.join(path_to_files,
                                       '*indresp.dta'))
    indresp = []
    prefixes = [] # for later
    for filename in all_files:
        try:
            prefix = filename.split('/')[-1][0:2]
            prefixes.append(prefix)
            colnames = [f'{prefix}{x}' for x in columns]
            temp_df = pd.read_stata(filename,
                           columns=['pidp'] + colnames)
            indresp.append(temp_df)
        except ValueError:
            prefix = filename.split('/')[-1][0:2]
            prefixes.append(prefix)
            colnames = [f'{prefix}{x}' for x in columns]
            all_cols = pd.read_stata(filename, convert_categoricals=False).columns
            idx2 = pd.Series(colnames).isin(all_cols)
            missing_cols = pd.Series(colnames)[~idx2].to_list()
            present_cols = pd.Series(colnames)[idx2].to_list()
            temp_df = pd.read_stata(filename,
                           columns=['pidp'] + present_cols)
            for col in missing_cols:
                temp_df[col] = np.nan
            indresp.append(temp_df[['pidp'] + colnames])
            
    for i, df in enumerate(indresp):
        df['wave'] = i+1
        df.columns = ['pidp'] + columns + ['wave'] # the order of the columns here is important
    out = pd.concat(indresp, ignore_index=True)
    out['max_waves'] = out.groupby('pidp')['wave'].transform('count').values
    return out


def load_data_h(path_to_files: str, columns: list):
    """
    This script only load data files scattered by waves in the folders.
    """
    all_files = glob.glob(os.path.join(path_to_files,
                                       '*hhresp.dta'))
    indresp = []
    prefixes = [] # for later
    for filename in all_files:
        try:
            prefix = filename.split('/')[-1][0:2]
            prefixes.append(prefix)
            colnames = [f'{prefix}{x}' for x in columns]
            temp_df = pd.read_stata(filename,
                           columns=colnames, convert_categoricals=False)
            indresp.append(temp_df)
        except ValueError:
            prefix = filename.split('/')[-1][0:2]
            prefixes.append(prefix)
            colnames = [f'{prefix}{x}' for x in columns]
            all_cols = pd.read_stata(filename, convert_categoricals=False).columns
            idx2 = pd.Series(colnames).isin(all_cols)
            missing_cols = pd.Series(colnames)[~idx2].to_list()
            present_cols = pd.Series(colnames)[idx2].to_list()
            temp_df = pd.read_stata(filename,
                           columns=present_cols)
            for col in missing_cols:
                temp_df[col] = np.nan
            indresp.append(temp_df[colnames])
            
    for i, df in enumerate(indresp):
        df['wave'] = i+1
        df.columns = columns + ['wave'] # the order of the columns here is important
    out = pd.concat(indresp, ignore_index=True)
    return out


def trimmer(x: pd.Series, lwb: float=0.0, upb: float=0.99):
    s = x.copy()
    lower_bound = s.quantile(lwb)
    upper_bound = s.quantile(upb)
    return s.apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)


def create_index(x):
    """
    Create relative index based on length of array.
    """
    y = np.arange(len(x)) + 1
    return y - x


def recoding_and_cleaning(in_data, cpih_data):
    """
    This function recodes and clean various variables
    Is in general a mess that runs once.
    """
    data = in_data.copy()
    data = data[~(data.pidp == 1020477375)] # Case with duplcates values
    data = data[~(data.pidp == 1156430447)] # Case with duplicate values
    data['sex_recoded'] = data.sex.str.strip().replace({
    'female': 0,
    'male': 1,
    "don't know": np.nan,
    'missing': np.nan,
    'refusal': np.nan})
    data['mastat_recoded'] = data.mastat_dv.str.strip().replace({
    'Married': 1,
    'Living as couple': 1,
    'In a registered same-sex civil partnership': 1,
    'Single and never married/in civil partnership': 0,
    'A former civil partner': 0,
    'Divorced': 0,
    'Separated but legally married': 0,
    'Widowed': 0,
    'Separated from civil partner': 0,
    'A surviving civil partner': 0,
    "don't know": np.nan,
    'missing': np.nan,
    'refusal': np.nan,
    'inapplicable': np.nan,
    'Child under 16': np.nan
    })
    data['ethn_8'] = data.ethn_dv.str.strip().replace({
    'british/english/scottish/welsh/northern irish': 'uk-white',
    'indian': 'indian',
    'pakistani': 'pakistani',
    'any other white background': 'white',
    'african': 'black',
    'bangladeshi': 'asian',
    'caribbean': 'black',
    'irish': 'white',
    'any other asian background': 'asian',
    'white and black caribbean': 'mixed',
    'any other ethnic group': 'other',
    'chinese': 'asian',
    'any other mixed background': 'mixed',
    'white and asian': 'mixed',
    'arab': 'other',
    'white and black african': 'mixed',
    'any other black background': 'black',
    'gypsy or irish traveller': 'white',
    'missing': np.nan
    })
    data['ethn_5'] = data.ethn_8.str.strip().replace({
    'uk-white': 'white',
    'indian': 'asian',
    'pakistani': 'asian'
    })
    data['edu'] = data.qfhigh_dv.str.strip().str.lower()
    data['edu_6'] = data.edu.str.strip().replace({
    'cse': 1,
    'other school cert': 1,
    'gcse/o level': 2,
    'standard/o/lower': 2,
    'a level': 3,
    'as level': 3,
    'highers (scot)': 3,
    'cert 6th year studies': 3,
    "i'nationl baccalaureate": 3,
    'welsh baccalaureate': 3,
    'diploma in he': 4,
    'nursing/other med qual': 4,
    'teaching qual not pgce': 4,
    '1st degree or equivalent': 5,
    'higher degree': 6,
    'other higher degree': 6,
    'none of the above': np.nan,
    'missing': np.nan,
    'inapplicable': np.nan,
    })
    data['edu_3'] = data.edu_6.replace({
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3
    })
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
    data['year'] = np.nan
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
    
    data['weight_yearx'] = np.nan
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==1), 'weight_yearx'] = data.indscus_xw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==1), 'weight_yearx'] = data.indscus_xw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==2), 'weight_yearx'] = data.indinus_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==2), 'weight_yearx'] = data.indinus_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==3), 'weight_yearx'] = data.indinub_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==3), 'weight_yearx'] = data.indinub_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==4), 'weight_yearx'] = data.indinub_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==4), 'weight_yearx'] = data.indinub_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==5), 'weight_yearx'] = data.indinub_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==5), 'weight_yearx'] = data.indinub_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==6), 'weight_yearx'] = data.indinub_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==6), 'weight_yearx'] = data.indinub_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==7), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==7), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==8), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==8), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==9), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==9), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==10), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==10), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==11), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==11), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=1)&(data.month_recoded<=12)&(data.wave==12), 'weight_yearx'] = data.indinui_lw
    data.loc[(data.month_recoded>=13)&(data.month_recoded<=24)&(data.wave==12), 'weight_yearx'] = data.indinui_lw
    
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
    data['ind_inc'] = data.fimnlabgrs_dv.replace({"don't know": np.nan,
                                              'missing': np.nan})
    data['ind_inc_deflated'] = (data['ind_inc'] / data['cpih']) * 100
    data['wage_h_deflated'] = (data['wage_h'] / data['cpih']) * 100
    data['hh_inc'] = data.fihhmngrs_dv.replace({-9: np.nan})
    data['hh_inc_deflated'] = (data['hh_inc'] / data['cpih']) * 100
    data['log_wage_h_deflated'] = np.log(data['wage_h_deflated'])
    data.loc[data.hh_inc_deflated < 0, 'hh_inc_deflated'] = np.nan
    data.loc[data.ind_inc_deflated < 0, 'ind_inc_deflated'] = np.nan

    #trimming
    data['hh_inc_deflated'] = trimmer(data['hh_inc_deflated'], )
    data['ind_inc_deflated'] = trimmer(data['ind_inc_deflated'])
    data['wage_h_deflated'] = trimmer(data['wage_h_deflated'])
    data['log_wage_h_deflated'] = trimmer(data['log_wage_h_deflated'])
    data['dvage'] = data.dvage.replace({
        'missing': np.nan,
        "don't know": np.nan,
        'refusal': np.nan,
        'inapplicable': np.nan})
    data['dvage'] = trimmer(data['dvage'])
    return data


def isc_data_preparation(data, conditions: dict):
    out = data.copy()
    employed = conditions['employed']
    dropna = conditions['dropna']
    target_var = conditions['target_var']
    min_treat_waves = conditions['min_treat_waves']
    min_waves_pretreat = conditions['min_waves_pretreat']
    if employed: # dropping if ever unemployed
        out['unemployed_bool'] = ~(out.employed == 'employed')
        out_copy = out.copy()
        to_drop = []
        for pidp in out.pidp.unique():
            temp_data = out[out.pidp==pidp].copy()
            if temp_data['unemployed_bool'].any(): #check if any time was unemployed
                to_drop.append(pidp) # adds to drop if above is true
        out = out_copy[~out_copy.pidp.isin(to_drop)].copy()
    if dropna: #dropping if ever missing in target var
        out_copy = out.copy()
        to_drop = []
        for pidp in out.pidp.unique():
            temp_data = out[out.pidp==pidp].copy()
            if temp_data[target_var].isnull().any():
                to_drop.append(pidp)
        out = out_copy[~out_copy.pidp.isin(to_drop)].copy()
    out['ever_treated'] = out.groupby('pidp')['treated'].transform(any).values
    out['year_reindex'] = out.sort_values(by=['pidp', 'year']).groupby('pidp').cumcount() + 1
    out.reset_index(drop=True, inplace=True)
    out.sort_values(by=['pidp', 'year_reindex'], inplace=True)
    out['year_treated'] = out.year[out.groupby('pidp')['treated'].transform('idxmax').values].values
    out['year_treat_reindex'] = out.year_reindex[out.groupby('pidp')['treated'].transform('idxmax').values].values
    out['initial_year'] = out.groupby('pidp')['year'].transform('min').values
    out['reindex'] = out.groupby('pidp')['year_treat_reindex'].transform(create_index).values
    out['years_treated'] = out.groupby('pidp')['treated'].transform('sum').values
    treated = out[out.ever_treated].copy()
    control = out[~out.ever_treated].copy()
    control['ever_treated'] = control.groupby('pidp')['treated'].transform(any)
    treated = treated[~(treated.years_treated < min_treat_waves)]
    treated = treated.drop(treated[(treated.year_treat_reindex < min_waves_pretreat)].index)
    return treated, control


def create_index(x):
    y = np.arange(len(x)) + 1
    return y - x

def create_relative_index(lst, point):
    index = lst.index(point)
    return [i - index for i in range(len(lst))]

def create_relative_MultiIndex(lst, point):
    index = lst.index(point)
    return [i - index for i in range(len(lst))]


# the expression to minimise, since \mu is 0, we are one looking for \omega/W
def argmin_w(W, Y_i, Y_0):
    return np.sqrt(np.sum((Y_0 - Y_i.dot(W))**2))

# a function wrapping the whole process
def get_w(Y_i, Y_0):
    w_start = [1/Y_i.shape[1]]*Y_i.shape[1]
    weights = minimize(partial(argmin_w, Y_i=Y_i, Y_0=Y_0),
                       np.array(w_start),
                       method='SLSQP',
                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), # constraint to sum to 1
                       bounds=[(0.0, 1.0)]*len(w_start),
                       )
    return weights.x

def is_consecutive(l):
    return sorted(l) == list(range(min(l), max(l)+1))
    idx = []
    for pidp in treated.pidp:
        idx.append(is_consecutive(treated[treated.pidp == pidp].wave))
    
def get_control_clean(c_data, t_data, features, target_var, weights=None):
    samples = []
    t_ids = t_data.pidp.unique().tolist()
    for t_id in t_ids:
        if t_data[t_data.pidp == t_id].shape[0] < 5:
            continue
        out = {}
        treat_time = t_data[t_data.pidp == t_id].year_treated.unique()[0]
        t_data = t_data.dropna(subset=['year']).copy()
        treat = t_data[t_data.pidp == t_id].pivot(index='pidp', columns='year')[features].T
        control = c_data.pivot(index='pidp', columns='year')[features].T
        sub_sample = pd.concat([treat, control], axis=1, join="inner") # concat-join-inner ensure using index (year) as key
        out['data'] = sub_sample.dropna(axis=1) # only complete columns
        out['treat_time'] = treat_time
        out['target_var'] = target_var
        out['weight'] = t_data[t_data.pidp == t_id][['year', weights]].set_index('year')
        samples.append(out)
    return samples

def sc(x, k_n):
    data = x['data'].copy()
    ncol = data.shape[1] - 1
    sample_weights = x['weight'].copy()
    data.index.names = ['var', 'year']
    t_time = x['treat_time']
    target_var = x['target_var']
    data.index = data.index.map(lambda idx: (idx[0], idx[1] - t_time))
    sample_weights.index = sample_weights.index - t_time
    data = data.sort_index(ascending=True).copy()
    data = data.loc[(slice(None), slice(-5, 5)), :].copy() # this limits to only -5 to 5 years
    df_T0 = data.loc[pd.IndexSlice[:, :-1], :]
    Y_0 = df_T0.iloc[:, 0].values
    if ncol < k_n:
        k_n = ncol
    try:
        kdt = KDTree(df_T0.T, leaf_size=30, metric='euclidean')
    except ValueError:
        return None
    idx = kdt.query(df_T0.T, k=k_n, return_distance=False)[0, 1:]
    Y_i = df_T0.iloc[:, idx].values
    weights = get_w(Y_i, Y_0)
    synth = data.iloc[:, idx].dot(weights).loc[target_var] # synthetic control is now based on the new subset of observations
    treated = data.iloc[:, 0].loc[target_var]
    diff = treated - synth
    weighted_diff = sample_weights.multiply(diff, axis=0)['weight_yearx']
    return {
        'synth': synth,
        'treated': treated,
        'diff': diff,
        'weighted_diff': weighted_diff
         }

def isc(data_objects: list, k_n: int=500) -> dict:
    synths = []
    treats = []
    diffs = []
    weighted_diffs = []
    with Pool() as p:
        out = p.starmap(sc, [(data, k_n) for data in data_objects])
    
    for ele in out:
        if ele is not None:
            synths.append(ele['synth'])
            treats.append(ele['treated'])
            diffs.append(ele['diff'])
            weighted_diffs.append(ele['weighted_diff'])
        else:
            continue
    return {'synths': synths,
            'treats': treats,
            'diffs': diffs,
            'weighted_diff': weighted_diffs}
