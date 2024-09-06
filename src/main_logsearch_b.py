import pandas as pd
import numpy as np
from utils import get_control_clean
from isc_lib import isc
import time
import warnings
import argparse
warnings.filterwarnings("ignore")


def get_intertisial_data(clean_t, clean_c, target_var):
    treated = pd.read_csv(clean_t, index_col=0)
    controls = pd.read_csv(clean_c, index_col=0)
    target_var = target_var
    samples = get_control_clean(controls, treated,
                                [target_var,
                                'dvage',
                                'mastat_recoded',
                                'sex_recoded',
                                'employed_num',
                                'hhsize',
                                'asian',
                                'black',
                                'mixed',
                                'other',
                                'low',
                                'middle'],
                                target_var,
                                'weight_yearx')
    return samples


def run_isc(clean_t, clean_c, target_var, out_suffix, k_n=35, placebo=False, seed=1):
    print(f'Getting data for {target_var}_{out_suffix}...')
    samples = get_intertisial_data(clean_t, clean_c, target_var)
    print('DONE')
    print(f'Running ISC for {target_var}_{out_suffix}...')
    start_time = time.time()
    out = isc(samples, penalized=True, reduction=True, k_n=k_n, placebo=placebo, seed=seed)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('DONE')
    print('Saving Data...')
    rmse = out['rmses']
    print('DONE')
    return np.mean(rmse), elapsed_time




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run ISC with log search over n and random seed")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility")

    args = parser.parse_args()
    fix_seed = int(args.seed)
    start_val = 5
    end_val = 500
    num_points = 30
    log_space = np.logspace(np.log2(start_val), np.log2(end_val), num=num_points, base=2)
    log_sequence = sorted(set(map(int, log_space)))
    N = []
    RMSEs = []
    times = []
    for n in log_sequence:
        rmse, e_time = run_isc('./data/byintensity/ii_t_hi.csv', './data/byintensity/ii_c_full.csv', 'ind_inc_deflated', 'hi', n, False, fix_seed)
        N.append(n)
        RMSEs.append(rmse)
        times.append(e_time)
    out = pd.DataFrame({
        'n': N,
        'rmse': RMSEs,
        'time': times
        })
    out.to_csv('./outputs/logsearch/logsearch_out_ii_500.csv')
