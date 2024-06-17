from sklearn.neighbors import KDTree
import random
import pandas as pd
import numpy as np
from toolz import partial
from scipy.optimize import minimize
import time
from tqdm import tqdm
import matplotlib
from multiprocessing import Pool, cpu_count
matplotlib.use("TkAgg")


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
    X = list(range(1, steps))
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
    new_columns = {}
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
        new_columns[f't{i}'] = t  # Store the new column data
#        T = pd.Series(t)
#        df[f't{i}'] = T
    new_columns_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_columns_df], axis=1)
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




def compute_rmspe(args):
    n_treat, n = args
    sim_df = gen_data_mix(n_treat=n_treat, range_gamma_pop_step=100)
    start = time.time()
    av_rmspe = simple_isc(sim_df, n, 50)
    end = time.time()
    calc_time = end - start
    return n_treat, n, av_rmspe, calc_time



#def test_fit(end_point, num_points):
#    """
#    Test the fit of the synthetic control model.
#
#    Args:
#        endpoint (int): The stopping number for our logarithmic range
#        num_points (int): The number of points in our list.
#
#    Returns:
#        DataFrame: Results of the fit tests including RMSPE and computation time.
#    """
#    rmspe_list = []
#    N = []
#    elapsed_times = []
#    n_treat_list = []
#    start = 10
#    end = end_point
#    num_points = num_points
#    log_space = np.logspace(np.log2(start), np.log2(end), num=num_points, base=2)
#    log_sequence = sorted(set(map(int, log_space)))
#
#    n_treat_values = [25, 50, 100, 250, 500]
#    total_iterations = len(n_treat_values) * len(log_sequence)
#
#    with tqdm(total=total_iterations) as pbar:
#        for n_treat in n_treat_values:
#            sim_df = gen_data_mix(n_treat=n_treat,
#                                  range_gamma_pop_step=100)
#            for n in log_sequence:
#                n_treat_list.append(n_treat)
#                start = time.time()
#                av_rmspe = simple_isc(sim_df, n, 50)
#                end = time.time()
#                calc_time = end - start
#                rmspe_list.append(av_rmspe)
#                N.append(n)
#                elapsed_times.append(calc_time)
#                pbar.update(1)
#
#    return pd.DataFrame({'N': N,
#                         'fit': rmspe_list,
#                         'time': elapsed_times,
#                         'n_treat': n_treat_list})


import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import os


def get_seed_list():
    seed_list_path = os.path.join(os.getcwd(), '..', 'data', 'seeds', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]



def process_single_task(args):
    n_treat, n, seed = args
    np.random.seed(seed)  # Set seed for reproducibility
    sim_df = gen_data_mix(n_treat=n_treat, range_gamma_pop_step=100)
    start = time.time()
    av_rmspe = simple_isc(sim_df, n, 50)
    end = time.time()
    calc_time = end - start
    return n_treat, n, av_rmspe, calc_time, seed


def test_fit(end_point, num_points):
    """
    Test the fit of the synthetic control model in parallel.

    Args:
        endpoint (int): The stopping number for our logarithmic range.
        num_points (int): The number of points in our list.

    Returns:
        DataFrame: Results of the fit tests including RMSPE, computation time, and seed used.
    """
    start_val = 10
    end_val = end_point
    log_space = np.logspace(np.log2(start_val), np.log2(end_val), num=num_points, base=2)
    log_sequence = sorted(set(map(int, log_space)))

    seed_limit = 10
    seed_list = get_seed_list()[:seed_limit]  # Assuming get_seed_list() returns a list of seeds

    n_treat_values = [25, 50, 100, 250]
    total_iterations = len(n_treat_values) * len(log_sequence) * len(seed_list)

    tasks = [(n_treat, n, seed) for seed in seed_list for n_treat in n_treat_values for n in log_sequence]

    results = []
    with tqdm(total=total_iterations) as pbar:
        with Pool(cpu_count()) as pool:
            for result in pool.imap_unordered(process_single_task, tasks):
                results.append(result)
                pbar.update()

    df_results = pd.DataFrame(results, columns=['N', 'fit', 'time', 'n_treat', 'seed'])
    return df_results


if __name__ == "__main__":
    fit_df = test_fit(1000, 50)
    fit_df.to_csv('../data/simulation/fit_data.csv')