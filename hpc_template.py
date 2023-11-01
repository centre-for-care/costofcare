import pandas as pd
from utils import get_control_clean
from utils import isc_b
import math
import pickle


def get_intertisial_data(clean_t, clean_c, target_var, out_suffix):
    treated = pd.read_csv(clean_t, index_col=0)
    controls = pd.read_csv(clean_c, index_col=0)
    print(len(treated.pidp.unique()))
    target_var = target_var
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
    with open(f'./outputs/{target_var}_{out_suffix}.pkl', 'wb') as file:
        pickle.dump(samples, file)
    return samples


def run_isc(clean_t, clean_c, target_var, out_suffix, k_n=35):
    samples = get_intertisial_data(clean_t, clean_c, target_var, out_suffix)
    out = isc_b(samples, penalized=False, custom_v='auto', reduction=True, k_n=k_n)

    diffs = pd.concat(out['diffs'], axis=1).sort_index()
    w_diffs = pd.concat(out['weighted_diff'], axis=1).sort_index()
    treats = pd.concat(out['treats'], axis=1).sort_index()
    synths = pd.concat(out['synths'], axis=1).sort_index()
    diffs.to_csv(f'./outputs/diffs_{target_var}_{out_suffix}.csv')
    w_diffs.to_csv(f'./outputs/w_diffs_{target_var}_{out_suffix}.csv')
    treats.to_csv(f'./outputs/treats_{target_var}_{out_suffix}.csv')
    synths.to_csv(f'./outputs/synths_{target_var}_{out_suffix}.csv')


def main():
    run_isc('./strata/ii_t_hi.csv', './strata/ii_c_full.csv', 'ind_inc_deflated', 'hi')
    run_isc('./strata/ii_t_li.csv', './strata/ii_c_full.csv', 'ind_inc_deflated', 'li')

    run_isc('./strata/hhi_t_hi.csv', './strata/hhi_c_full.csv', 'hh_inc_deflated', 'hi')
    run_isc('./strata/hhi_t_li.csv', './strata/hhi_c_full.csv', 'hh_inc_deflated', 'li')

    run_isc('./strata/is_t_hi.csv', './strata/is_c_full.csv', 'inc_share', 'hi')
    run_isc('./strata/is_t_li.csv', './strata/is_c_full.csv', 'inc_share', 'li')

    run_isc('./strata/pe_t_hi.csv', './strata/pe_c_full.csv', 'prob_emp', 'hi')
    run_isc('./strata/pe_t_li.csv', './strata/pe_c_full.csv', 'prob_emp', 'li')

    target_var = 'ind_inc_deflated'
    prefix  = 'ii'

    run_isc(f'./strata/{prefix}_t_hi_m.csv', f'./strata/{prefix}_c_m.csv', target_var, 'hi_m')
    run_isc(f'./strata/{prefix}_t_hi_f.csv', f'./strata/{prefix}_c_f.csv', target_var, 'hi_f')

    run_isc(f'./strata/{prefix}_t_hi_w.csv', f'./strata/{prefix}_c_w.csv', target_var, 'hi_w')
    run_isc(f'./strata/{prefix}_t_hi_nw.csv', f'./strata/{prefix}_c_nw.csv', target_var, 'hi_nw')

    run_isc(f'./strata/{prefix}_t_hi_edl.csv', f'./strata/{prefix}_c_edl.csv', target_var, 'hi_edl')
    run_isc(f'./strata/{prefix}_t_hi_edm.csv', f'./strata/{prefix}_c_edm.csv', target_var, 'hi_edm')
    run_isc(f'./strata/{prefix}_t_hi_edh.csv', f'./strata/{prefix}_c_edh.csv', target_var, 'hi_edh')

    target_var = 'hh_inc_deflated'
    prefix  = 'hhi'
    run_isc(f'./strata/{prefix}_t_hi_m.csv', f'./strata/{prefix}_c_m.csv', target_var, 'hi_m')
    run_isc(f'./strata/{prefix}_t_hi_f.csv', f'./strata/{prefix}_c_f.csv', target_var, 'hi_f')

    run_isc(f'./strata/{prefix}_t_hi_w.csv', f'./strata/{prefix}_c_w.csv', target_var, 'hi_w')
    run_isc(f'./strata/{prefix}_t_hi_nw.csv', f'./strata/{prefix}_c_nw.csv', target_var, 'hi_nw')

    run_isc(f'./strata/{prefix}_t_hi_edl.csv', f'./strata/{prefix}_c_edl.csv', target_var, 'hi_edl')
    run_isc(f'./strata/{prefix}_t_hi_edm.csv', f'./strata/{prefix}_c_edm.csv', target_var, 'hi_edm')
    run_isc(f'./strata/{prefix}_t_hi_edh.csv', f'./strata/{prefix}_c_edh.csv', target_var, 'hi_edh')

    target_var = 'inc_share'
    prefix  = 'is'
    run_isc(f'./strata/{prefix}_t_hi_m.csv', f'./strata/{prefix}_c_m.csv', target_var, 'hi_m')
    run_isc(f'./strata/{prefix}_t_hi_f.csv', f'./strata/{prefix}_c_f.csv', target_var, 'hi_f')

    run_isc(f'./strata/{prefix}_t_hi_w.csv', f'./strata/{prefix}_c_w.csv', target_var, 'hi_w')
    run_isc(f'./strata/{prefix}_t_hi_nw.csv', f'./strata/{prefix}_c_nw.csv', target_var, 'hi_nw')

    run_isc(f'./strata/{prefix}_t_hi_edl.csv', f'./strata/{prefix}_c_edl.csv', target_var, 'hi_edl')
    run_isc(f'./strata/{prefix}_t_hi_edm.csv', f'./strata/{prefix}_c_edm.csv', target_var, 'hi_edm')
    run_isc(f'./strata/{prefix}_t_hi_edh.csv', f'./strata/{prefix}_c_edh.csv', target_var, 'hi_edh')

    target_var = 'prob_emp'
    prefix  = 'pe'
    run_isc(f'./strata/{prefix}_t_hi_m.csv', f'./strata/{prefix}_c_m.csv', target_var, 'hi_m')
    run_isc(f'./strata/{prefix}_t_hi_f.csv', f'./strata/{prefix}_c_f.csv', target_var, 'hi_f')

    run_isc(f'./strata/{prefix}_t_hi_w.csv', f'./strata/{prefix}_c_w.csv', target_var, 'hi_w')
    run_isc(f'./strata/{prefix}_t_hi_nw.csv', f'./strata/{prefix}_c_nw.csv', target_var, 'hi_nw')

    run_isc(f'./strata/{prefix}_t_hi_edl.csv', f'./strata/{prefix}_c_edl.csv', target_var, 'hi_edl')
    run_isc(f'./strata/{prefix}_t_hi_edm.csv', f'./strata/{prefix}_c_edm.csv', target_var, 'hi_edm')
    run_isc(f'./strata/{prefix}_t_hi_edh.csv', f'./strata/{prefix}_c_edh.csv', target_var, 'hi_edh')


if __name__ == "__main__":
    main()
