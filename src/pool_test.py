import pandas as pd
from utils import get_control_clean
from utils import isc_b
import numpy as np


def run_isc(clean_t, clean_c, target_var):
    print('Getting data...')
    samples = get_intertisial_data(clean_t, clean_c, target_var)
    print('Data loaded')
    print('Running ISC...')
    for s_size in np.logspace(0.5, 2, num=15, base=10,dtype='int'):
        out = isc_b(samples, penalized=True, custom_v='auto', reduction=True, k_n=s_size)
        diffs = pd.concat(out['diffs'], axis=1).sort_index()
        diffs.to_csv(f'./pooltest/diffs_{target_var}_{s_size}_li.csv')
    print('Outputs saved...')


def get_intertisial_data(clean_t, clean_c, target_var):
    treated = pd.read_csv(clean_t, index_col=0)
    controls = pd.read_csv(clean_c, index_col=0)
    samples = get_control_clean(controls, treated,
                                [target_var,
                                'dvage',
                                'mastat_recoded',
                                'asian',
                                'black',
                                'mixed',
                                'other',
                                'low',
                                'middle'],
                                target_var,
                                'weight_yearx')
    return samples


def main():
    run_isc('./strata/ii_t_li.csv', './strata/ii_c_full.csv', 'ind_inc_deflated')
    run_isc('./strata/hhi_t_li.csv', './strata/hhi_c_full.csv', 'hh_inc_deflated')
    run_isc('./strata/is_t_li.csv', './strata/is_c_full.csv', 'inc_share')
    run_isc('./strata/pe_t_li.csv', './strata/pe_c_full.csv', 'prob_emp')

if __name__ == "__main__":
    main()
