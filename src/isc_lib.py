import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from multiprocessing import Pool
from pysyncon import Dataprep, Synth, PenalizedSynth
import warnings
import tqdm 
warnings.filterwarnings("ignore", category=FutureWarning)


def sc(data_object, 
         penalized: bool=False, 
         reduction: bool=False, 
         k_n: int=500, 
         lambda_: float=.01,
         ):
    data = data_object['data'].copy()
    ncol = data.shape[1] - 1
    sample_weights = data_object['weight'].copy()
    data.index.names = ['var', 'year']
    t_time = data_object['treat_time']
    target_var = data_object['target_var']
    treated_unit = data_object['treat_id']
    data.index = data.index.map(lambda idx: (idx[0], idx[1] - t_time))
    sample_weights.index = sample_weights.index - t_time
    data = data.sort_index(ascending=True).copy()
    data = data.loc[(slice(None), slice(-8, 8)), :].copy()
    min_year = data.index.get_level_values('year').min().astype(int)
    max_year = data.index.get_level_values('year').max().astype(int)
    if reduction:
        if ncol < k_n:
            k_n = ncol
        try:
            df_T0 = data.loc[pd.IndexSlice[:, :-1], :]
            kdt = KDTree(df_T0.T, leaf_size=30, metric='euclidean')
            idx = kdt.query(df_T0.T, k=k_n, return_distance=False)[0, :]
            data = data.iloc[:, idx]
        except ValueError:
            return None
    melted_df = data.reset_index().melt(id_vars=['var', 'year'], var_name='pidp')
    pivoted_df = melted_df.pivot(index=['year', 'pidp'], columns='var', values='value').reset_index()
    pivoted_df =pivoted_df.sort_values(by=['pidp', 'year'])
    pivoted_df = pivoted_df.reset_index(drop=True)
    pivoted_df.columns.name = ''
    controls = pivoted_df.pidp[pivoted_df.pidp!=treated_unit].unique().tolist()
    covariates = pivoted_df.columns.to_list()
    covariates.remove('year')
    covariates.remove('pidp')
    print(pivoted_df)
    try:
        dataprep = Dataprep(
            foo=pivoted_df,
            predictors=covariates,
            predictors_op="mean",
            time_predictors_prior=range(min_year, 0),
            dependent=target_var,
            unit_variable="pidp",
            time_variable="year",
            treatment_identifier=treated_unit,
            controls_identifier=controls,
            time_optimize_ssr=range(min_year, 0),
        )
    except ValueError:
        return None
    try:
        if penalized:
            synth = PenalizedSynth()
            synth.fit(dataprep=dataprep, lambda_=lambda_)
        else:
            synth = Synth()
            synth.fit(dataprep)
    except KeyError:
        return None
    
    # bootstrapping section
    av_att_ind = []
    for _ in range(50):
        n = len(controls)
        sample_controls = np.random.choice(controls, n, replace=True).tolist()
        try:
            temp_dataprep = Dataprep(
            foo=pivoted_df,
            predictors=covariates,
            predictors_op="mean",
            time_predictors_prior=range(min_year, 0),
            dependent=target_var,
            unit_variable="pidp",
            time_variable="year",
            treatment_identifier=treated_unit,
            controls_identifier=sample_controls,
            time_optimize_ssr=range(min_year, 0),
        )
        except ValueError:
            return None
        try:
            if penalized:
                temp_synth = PenalizedSynth()
                temp_synth.fit(dataprep=temp_dataprep, lambda_=lambda_)
            else:
                temp_synth = Synth()
                temp_synth.fit(dataprep=temp_dataprep)
        except KeyError:
            return None
        temp_s_cntrl = temp_synth._synthetic(Z0=temp_synth.dataprep.make_outcome_mats(time_period=range(min_year,max_year+1))[0])# synthetic control is now based on the new subset of observations
        temp_treated = data[treated_unit].loc[target_var]
        temp_diff = temp_treated - temp_s_cntrl
        av_att_ind.append(temp_diff)
    boots_var = pd.concat(av_att_ind, axis=1).var(axis=1)
    s_cntrl = synth._synthetic(Z0=synth.dataprep.make_outcome_mats(time_period=range(min_year,max_year+1))[0])# synthetic control is now based on the new subset of observations
    treated = data[treated_unit].loc[target_var]
    diff = treated - s_cntrl
    rmse = np.sqrt(np.mean(np.square(diff[:-1])))
    w_diff = sample_weights.multiply(diff, axis=0)['weight_yearx']
    w_treated = sample_weights.multiply(treated, axis=0)['weight_yearx']
    w_synth = sample_weights.multiply(s_cntrl, axis=0)['weight_yearx']
    return {
        'rmse': rmse,
        'synth': s_cntrl,
        'treated': treated,
        'diff': diff,
        'w_diff': w_diff,
        'w_treated': w_treated,
        'w_synth': w_synth,
        'boots_var': boots_var
         }


def isc(data_objects: list, 
          penalized: bool=False, 
          reduction: bool=False, 
          k_n: int=500, 
          lambda_: float=.01) -> dict:
    synths = []
    treats = []
    diffs = []
    w_diffs = []
    w_treats = []
    w_synths = []
    rmses = []
    boots_vars = []
    with Pool() as p:
        out = p.starmap(sc, tqdm.tqdm([(data, penalized, reduction, k_n, lambda_) for data in data_objects]))
    
    for ele in out:
        if ele is not None:
            rmses.append(ele['rmse'])
            synths.append(ele['synth'])
            treats.append(ele['treated'])
            diffs.append(ele['diff'])
            w_diffs.append(ele['w_diff'])
            w_treats.append(ele['w_treated'])
            w_synths.append(ele['w_synth'])
            boots_vars.append(ele['boots_var'])
        else:
            continue
    return {'rmses': rmses,
            'synths': synths,
            'treats': treats,
            'diffs': diffs,
            'w_diffs': w_diffs,
            'w_treats': w_treats,
            'w_synths': w_synths,
            'boots_vars': boots_vars}

