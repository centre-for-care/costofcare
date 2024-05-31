from sklearn.neighbors import KDTree
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolz import partial
from scipy.optimize import minimize
import time


def argmin_w(W, Y_i, Y_0):
    """
    Minimize the expression for synthetic control weights.
    
    Args:
        W (array): Array of weights.
        Y_i (array): Matrix of control unit outcomes.
        Y_0 (array): Vector of treated unit outcomes.
    
    Returns:
        float: The minimized value of the synthetic control expression.
    """
    return np.sqrt(np.sum((Y_0 - Y_i.dot(W))**2))


def get_w(Y_i, Y_0):
    """
    Get the synthetic control weights.
    
    Args:
        Y_i (array): Matrix of control unit outcomes.
        Y_0 (array): Vector of treated unit outcomes.
    
    Returns:
        array: Optimal weights for the synthetic control.
    """
    w_start = [1/Y_i.shape[1]]*Y_i.shape[1]
    weights = minimize(partial(argmin_w, Y_i=Y_i, Y_0=Y_0),
                       np.array(w_start),
                       method='SLSQP',
                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), # constraint to sum to 1
                       bounds=[(0.0, 1.0)]*len(w_start),
                       )
    return weights.x


def gen_treated(mu, gamma, alpha, beta, steps, treat_time):
    """
    Generate data for treated units.
    
    Args:
        mu (float): Mean of the normal distribution.
        gamma (float): Standard deviation of the normal distribution.
        alpha (float): Alpha parameter for the gamma distribution.
        beta (float): Beta parameter for the gamma distribution.
        steps (int): Number of time steps.
        treat_time (int): Time step at which treatment begins.
    
    Returns:
        list: Generated data for the treated unit.
    """
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
    """
    Generate a mixture of control and treated data.
    
    Args:
        steps (int): Number of time steps.
        cases (int): Number of control cases.
        pop_n (int): Number of subpopulations.
        range_treat_alpha (int): Range for alpha parameter in gamma distribution.
        range_treat_beta (int): Range for beta parameter in gamma distribution.
        n_treat (int): Number of treated cases.
        treat_time (int): Time step at which treatment begins.
        range_gamma_pop (int): Range for gamma parameter in normal distribution.
        range_gamma_pop_step (int): Step size for gamma parameter in normal distribution.
        range_mu_pop (int): Range for mu parameter in normal distribution.
        range_mu_pop_step (int): Step size for mu parameter in normal distribution.
    
    Returns:
        DataFrame: Generated data for control and treated units.
    """
    X = list(range(1,steps))
    c = []
    # Parameter of the subpopulations
    mu_list = random.choices(range(0,range_gamma_pop, range_gamma_pop_step), k=pop_n)
    gamma_list = random.choices(range(0,range_mu_pop, range_mu_pop_step), k=pop_n)
    # Loop generating cases in each subpopulation
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
    # Loop generating data for the treated case/cases
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
    """
    Perform simple synthetic control analysis.
    
    Args:
        data (DataFrame): Data containing control and treated units.
        n (int): Number of nearest neighbors to consider.
        t_0 (int): Time step at which treatment begins.
    
    Returns:
        float: Average root mean square prediction error (RMSPE).
    """
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
    """
    Test the fit of the synthetic control model.
    
    Args:
        n (int): Maximum number of nearest neighbors to consider.
        module_step (int): Step size for varying the number of nearest neighbors.
    
    Returns:
        DataFrame: Results of the fit tests including RMSPE and computation time.
    """
    N = []
    rmspe_list = []
    elapsed_times = []
    sim_df = gen_data_mix(n_treat=70,
                          range_gamma_pop_step=100)
    for n in range(1,n):
        if n % module_step == 0:
            start = time.time()
            av_rmspe = simple_isc(sim_df, n, 50)
            end = time.time()
            calc_time = end - start
            rmspe_list.append(av_rmspe)
            N.append(n)
            elapsed_times.append(calc_time)
    return pd.DataFrame({'N':N,
                         'fit': rmspe_list,
                         'time': elapsed_times})


if __name__ == "__main__":
    fit_df = test_fit(200, 50)
    fit_df.plot(x='N', y='fit', legend=False, figsize=(14, 7))
    plt.xlabel('Donor Pool Sample Size')
    plt.ylabel('Root Mean Square Prediction Error')
    plt.show()