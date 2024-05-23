import pandas as pd
from pysyncon import Dataprep, Synth, PenalizedSynth
import numpy as np


def with_effect(df,
                treated_unit,
                null_hypothesis,
                start_at,
                window):
    window_mask = (df.index >= start_at) & (df.index < (start_at +window))
    y = np.where(window_mask, df[treated_unit] - null_hypothesis, df[treated_unit])
    return df.assign(**{treated_unit: y})


def residuals(df_aug,
              treated_unit,
              null,
              intervention_start,
              window,
              time_var,
              unit_var,
              dependent,
              controls):
    null_data = with_effect(df_aug, treated_unit, null, intervention_start, window)
    null_data_m = null_data.reset_index().melt(id_vars=time_var, var_name=unit_var, value_name=dependent)
    dataprep = Dataprep(
            foo=null_data_m,
            predictors=[dependent],
            predictors_op="mean",
            time_predictors_prior=range(1970, intervention_start),
            dependent=dependent,
            unit_variable=unit_var,
            time_variable=time_var,
            treatment_identifier=treated_unit,
            controls_identifier=controls,
            time_optimize_ssr=range(1970, intervention_start)
            )
    synth = PenalizedSynth()
    synth.fit(dataprep)
    y0_est = synth._synthetic(Z0=synth.dataprep.make_outcome_mats(time_period=range(1970,intervention_start+window+1))[0])
    print(y0_est)
    residuals = null_data[treated_unit] - y0_est
    print(null_data[treated_unit])
    test_mask = (null_data.index >= intervention_start) & (null_data.index < (intervention_start + window))
    out = pd.DataFrame({
        "y0": null_data[treated_unit],
        "y0_est": y0_est,
        "residuals": residuals,
        "post_intervention": test_mask})[lambda d: d.index < (intervention_start + window)]
    return out


def test_statistic(u_hat,
                   q=1,
                   axis=0):
    return (np.abs(u_hat) ** q).mean(axis=axis) ** (1/q)


def p_value(resid_df,
            q=1):
    u = resid_df["residuals"].values
    post_intervention = resid_df["post_intervention"].values
    block_permutations = np.stack([np.roll(u, permutation, axis=0)[post_intervention]
                                   for permutation in range(len(u))])
    statistics = test_statistic(block_permutations, q=1, axis=1)
    p_val = np.mean(statistics >= statistics[0])
    return p_val


def p_val_grid(df_p,
               treated_unit,
               nulls,
               intervention_start,
               period,
               time_var,
               unit_var,
               dependent,
               controls):
    df_aug = df_p[df_p.index < intervention_start]._append(df_p.loc[period])
    p_vals =  {null: p_value(residuals(df_aug,
                                       treated_unit,
                                       null=null,
                                       intervention_start=period,
                                       window=1,
                                       time_var=time_var,
                                       unit_var=unit_var,
                                       dependent=dependent,
                                       controls=controls
                                       )) for null in nulls}
    return pd.DataFrame(p_vals, index=[period]).T


def confidence_interval_from_p_values(p_values,
                                      alpha=0.1):
    big_p_values = p_values[p_values.values >= alpha]
    return pd.DataFrame({
        f"{int(100-alpha*100)}_ci_lower": big_p_values.index.min(),
        f"{int(100-alpha*100)}_ci_upper": big_p_values.index.max(),
    }, index=[p_values.columns[0]])


def compute_period_ci(df_p,
                      treated_unit,
                      nulls,
                      intervention_start,
                      period,
                      time_var,
                      unit_var,
                      dependent,
                      controls,
                      alpha=0.1):
    p_vals = p_val_grid(df_p=df_p,
                        treated_unit=treated_unit,
                        nulls=nulls,
                        intervention_start=intervention_start,
                        period=period,
                        time_var=time_var,
                        unit_var=unit_var,
                        dependent=dependent,
                        controls=controls)
    return confidence_interval_from_p_values(p_vals, alpha=alpha)


def confidence_interval(df,
                        treated_unit,
                        nulls,
                        intervention_start,
                        window,
                        time_var,
                        unit_var,
                        dependent,
                        controls,
                        alpha=0.05):
    df_p = df.pivot(index=time_var, columns=unit_var, values=dependent)
    return pd.concat([compute_period_ci(df_p,
                                        treated_unit,
                                        nulls,
                                        intervention_start,
                                        period,
                                        time_var,
                                        unit_var,
                                        dependent,
                                        controls,
                                        alpha) 
                      for period in range(intervention_start, intervention_start+window)])




data = pd.read_csv("data/smoking.csv")

controls = data.state[data.state!='California'].unique().tolist()

nulls = np.linspace(-60, 20, 100)

ci_df = confidence_interval(
    data,
    "California",
    nulls=nulls,
    intervention_start=1988,
    window=2000 - 1988 + 1,
    time_var='year',
    unit_var='state',
    dependent='cigsale',
    controls=controls
)

print(ci_df)