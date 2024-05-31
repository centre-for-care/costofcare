from os import replace
from sklearn.neighbors import KDTree
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolz import partial
from scipy.optimize import minimize


# the expression to minimise, since \mu is 0, we are one looking for \omega/W
def argmin_w(W, Y_i, Y_0):
    return np.sqrt(np.sum((Y_0 - Y_i.dot(W))**2))
    
# a very simple version of synth controls
def get_w(Y_i, Y_0):
    w_start = [1/Y_i.shape[1]]*Y_i.shape[1]
    weights = minimize(partial(argmin_w, Y_i=Y_i, Y_0=Y_0),
                       np.array(w_start),
                       method='SLSQP',
                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), # constraint to sum to 1
                       bounds=[(0.0, 1.0)]*len(w_start),
                       )
    return weights.x


def gen_treated(mu, gamma, alpha, beta, steps, treat_time):
    t = []
    py = random.gauss(mu, gamma)
    for i in range(1,steps):
        if i <= treat_time:
            yi = py + random.gauss(0, gamma)
            py = yi
        else:
            yi = py - random.gammavariate(alpha, beta)
            py = yi
        t.append(yi)
    return t


def gen_data_mix(steps=100,
                 cases=1000,
                 pop_n=10,
                 range_treat_alpha=2,
                 range_treat_beta=5,
                 n_treat=100,
                 treat_time=50,
                 range_gamma_pop=500,
                 range_gamma_pop_step=10,
                 range_mu_pop=10,
                 range_mu_pop_step=1,
                 ):
    X = list(range(1,steps))
    c = []
    #Parameter of the subpopulations
    mu_list = random.choices(range(0,range_gamma_pop, range_gamma_pop_step), k=pop_n)
    gamma_list = random.choices(range(0,range_mu_pop, range_mu_pop_step), k=pop_n)
    #Loop generating cases in each subpopulation
    for i in range(0, cases):
        y = []
        mu = random.choice(mu_list)
        gamma = random.choice(gamma_list)
        py = random.gauss(mu, gamma)
        for x in X:
            yi = py + random.gauss(0, gamma)
            py = yi
            y.append(yi)
        c.append(y)
    df = pd.DataFrame(c).T
    df.columns = [f'c{x}' for x in range(1, cases+1)]
    #loop generating data for the treated case/cases
    alpha_list = random.choices(range(1,range_treat_alpha), k=n_treat)
    beta_list = random.choices(range(1,range_treat_beta), k=n_treat)
    for i in range(0, n_treat):
        mu = random.choice(mu_list)
        gamma = random.choice(gamma_list)
        alpha = random.choice(alpha_list)
        beta = random.choice(beta_list)
        t = gen_treated(mu=mu,
                        gamma=gamma,
                        alpha=alpha,
                        beta=beta,
                        steps=steps,
                        treat_time=treat_time)
        T = pd.Series(t)
        df[f't{i}'] = T
    return df


def simple_isc(data, n, t_0):
    control_cols = [col for col in data.columns if col.startswith('c')]
    treated_cols = [col for col in data.columns if col.startswith('t')]
    controls = data[control_cols]
    treated = data[treated_cols]
    errors = []
    for column in treated:
        df = pd.concat([controls, treated[column]], axis=1)
        df_T0 = df.loc[:t_0, :].copy()
        Y_0 = df_T0[column].values
        kdt = KDTree(df_T0.T, leaf_size=30, metric='euclidean')
        idx_full = kdt.query(df_T0.T, k=n, return_distance=False)
        idx = idx_full[-1, 1:]
        Y_i = df_T0.iloc[:, idx]
        weights = get_w(Y_i, Y_0)
        synth = controls.iloc[:, idx].dot(weights)
        rmspe = np.sqrt(np.mean((Y_0[:t_0] - synth[:t_0])**2))
        errors.append(rmspe)
    return np.mean(errors)



def test_fit(n, module_step):
    N = []
    rmspe_list = []
    sim_df = gen_data_mix(n_treat=70,
                          range_gamma_pop_step=100)
    for n in range(1,n):
        if n % module_step == 0:
            av_rmspe = simple_isc(sim_df, n, 50)
            rmspe_list.append(av_rmspe)
            N.append(n)
    return pd.DataFrame({'N':N, 'fit': rmspe_list})


fit_df = test_fit(100, 2)

fit_df.plot(x='N', y='fit', legend=False, figsize=(14, 7))
plt.xlabel('Donor Pool Sample Size')
plt.ylabel('Root Mean Square Prediction Error')
plt.show()