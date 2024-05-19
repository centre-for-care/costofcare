import pandas as pd
from pysyncon import Dataprep, Synth
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("data/smoking.csv")

controls = data.state[data.state!='California'].unique().tolist()

dataprep = Dataprep(
            foo=data,
            predictors=['cigsale'],
            predictors_op="mean",
            time_predictors_prior=range(1970, 1988),
            dependent="cigsale",
            unit_variable="state",
            time_variable="year",
            treatment_identifier='California',
            controls_identifier=controls,
            time_optimize_ssr=range(1970, 1988)
            )

synth = Synth()
synth.fit(dataprep)

synth.path_plot(time_period=range(1970, 2000), treatment_time=1988)

data_p = data.pivot(index="year", columns="state", values="cigsale")

synth.dataprep.make_outcome_mats(time_period=range(1970,2000+1))[0]

#synth_vals = data_p.drop(columns="California").dot(synth.W)
synth_vals = synth._synthetic(Z0=synth.dataprep.make_outcome_mats(time_period=range(1970,2000+1))[0])

plt.plot(data_p["California"], label="California")
plt.plot(data_p["California"].index, synth_vals, label="SC")
plt.vlines(x=1988, ymin=40, ymax=120, linestyle=":", lw=2, label="Proposition 99", color="black")
plt.legend()
plt.show()


pred_data = data_p.assign(**{"residuals": data_p["California"] - synth._synthetic(range(1970, 2001))})


def with_effect(df, state, null_hypothesis, start_at, window):
    window_mask = (df.index >= start_at) & (df.index < (start_at +window))
    
    y = np.where(window_mask, df[state] - null_hypothesis, df[state])
    
    return df.assign(**{state: y})


def residuals(df, state, null, intervention_start, window):
    
    null_data = with_effect(df, state, null, intervention_start, window)
    null_data_m = null_data.reset_index().melt(id_vars=['year'], var_name='state')
            
    dataprep = Dataprep(
            foo=null_data_m,
            predictors=['value'],
            predictors_op="mean",
            time_predictors_prior=range(1970, 1988),
            dependent="value",
            unit_variable="state",
            time_variable="year",
            treatment_identifier=state,
            controls_identifier=controls,
            time_optimize_ssr=range(1970, 1988)
            )

    synth = PenalizedSynth()
    synth.fit(dataprep, lambda_=0.1)
    
    y0_est = synth._synthetic(range(1970, 2001))
    
    residuals = null_data[state] - y0_est
    
    test_mask = (null_data.index >= intervention_start) & (null_data.index < (intervention_start + window))
    
    return pd.DataFrame({
        "y0": null_data[state],
        "y0_est": y0_est,
        "residuals": residuals,
        "post_intervention": test_mask})[lambda d: d.index < (intervention_start + window)]


residuals_df = residuals(data_p,
                         "California",
                         null=0.0,
                         intervention_start=1988,
                         window=2000-1988+1)

residuals_df.head()


def test_statistic(u_hat, q=1, axis=0):
    return (np.abs(u_hat) ** q).mean(axis=axis) ** (1/q)


print("H0:0 ", test_statistic(residuals_df.query("post_intervention")["residuals"]))


def p_value(resid_df, q=1):
    
    u = resid_df["residuals"].values
    post_intervention = resid_df["post_intervention"].values
    
    block_permutations = np.stack([np.roll(u, permutation, axis=0)[post_intervention]
                                   for permutation in range(len(u))])
    
    statistics = test_statistic(block_permutations, q=1, axis=1)
    
    p_val = np.mean(statistics >= statistics[0])

    return p_val

p_value(residuals_df)


def p_val_grid(df, state, nulls, intervention_start, period):
    
    df_aug = df[df.index < intervention_start]._append(df.loc[period])
    
    p_vals =  {null: p_value(residuals(df_aug,
                                       state,
                                       null=null,
                                       intervention_start=period,
                                       window=1)) for null in nulls}        
        
    return pd.DataFrame(p_vals, index=[period]).T



def confidence_interval_from_p_values(p_values, alpha=0.1):
    big_p_values = p_values[p_values.values >= alpha]
    return pd.DataFrame({
        f"{int(100-alpha*100)}_ci_lower": big_p_values.index.min(),
        f"{int(100-alpha*100)}_ci_upper": big_p_values.index.max(),
    }, index=[p_values.columns[0]])


def compute_period_ci(df, state, nulls, intervention_start, period, alpha=0.1):
    p_vals = p_val_grid(df=df,
                        state=state,
                        nulls=nulls,
                        intervention_start=intervention_start,
                        period=period)
    
    return confidence_interval_from_p_values(p_vals, alpha=alpha)


def confidence_interval(df, state, nulls, intervention_start, window, alpha=0.1, jobs=4):    
    return pd.concat([compute_period_ci(df, state, nulls, intervention_start, period, alpha) 
                      for period in range(intervention_start, intervention_start+window)])

nulls = np.linspace(-60, 20, 100)

ci_df = confidence_interval(
    data_p,
    "California",
    nulls=nulls,
    intervention_start=1988,
    window=2000 - 1988 + 1
)

pred_data = data_p.assign(**{"residuals": data_p["California"] - synth._synthetic(range(1970, 2001))})


plt.figure(figsize=(10,5))
plt.fill_between(ci_df.index, ci_df["90_ci_lower"], ci_df["90_ci_upper"], alpha=0.2,  color="C1")
plt.plot(pred_data["California"].index, pred_data["residuals"], label="California", color="C1")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=2, color="Black")
plt.vlines(x=1988, ymin=10, ymax=-50, linestyle=":", color="Black", lw=2, label="Proposition 99")
plt.legend()
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.show()

