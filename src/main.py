import pandas as pd
from utils import get_control_clean
from isc_lib import isc


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


def run_isc(clean_t, clean_c, target_var, out_suffix, k_n=35):
    print(f'Getting data for {target_var}_{out_suffix}...')
    samples = get_intertisial_data(clean_t, clean_c, target_var)
    print('DONE')
    print(f'Running ISC for {target_var}_{out_suffix}...')
    out = isc(samples, penalized=True, reduction=True, k_n=k_n)
    print('DONE')
    print('Saving Data...')
    diffs = pd.concat(out['diffs'], axis=1).sort_index()
    w_diffs = pd.concat(out['w_diff'], axis=1).sort_index()
    treats = pd.concat(out['treats'], axis=1).sort_index()
    w_treats = pd.concat(out['w_treats'], axis=1).sort_index()
    synths = pd.concat(out['synths'], axis=1).sort_index()
    w_synths = pd.concat(out['w_synths'], axis=1).sort_index()
    boot_vars = pd.concat(out['boots_vars'], axis=1).mean(axis=1).sort_index()
    diffs.to_csv(f'../outputs/diffs_{target_var}_{out_suffix}.csv')
    w_diffs.to_csv(f'../outputs/w_diffs_{target_var}_{out_suffix}.csv')
    treats.to_csv(f'../outputs/treats_{target_var}_{out_suffix}.csv')
    w_treats.to_csv(f'../outputs/w_treats_{target_var}_{out_suffix}.csv')
    synths.to_csv(f'../outputs/synths_{target_var}_{out_suffix}.csv')
    w_synths.to_csv(f'../outputs/w_synths_{target_var}_{out_suffix}.csv')
    boot_vars.to_csv(f'../outputs/boot_vars_{target_var}_{out_suffix}.csv')
    print('DONE')
    return 0


if __name__ == "__main__":
    run_isc('../data/byintensity/ii_t_hi.csv', '../data/byintensity/ii_c_full.csv', 'ind_inc_deflated', 'hi', 10)
    run_isc('../data/byintensity/ii_t_mhi.csv', '../data/byintensity/ii_c_full.csv', 'ind_inc_deflated', 'mhi', 10)
    run_isc('../data/byintensity/ii_t_mli.csv', '../data/byintensity/ii_c_full.csv', 'ind_inc_deflated', 'mli', 10)
    run_isc('../data/byintensity/ii_t_li.csv', '../data/byintensity/ii_c_full.csv', 'ind_inc_deflated', 'li', 10)

    run_isc('../data/byintensity/hhi_t_hi.csv', '../data/byintensity/hhi_c_full.csv', 'hh_inc_deflated', 'hi', 10)
    run_isc('../data/byintensity/hhi_t_mhi.csv', '../data/byintensity/hhi_c_full.csv', 'hh_inc_deflated', 'mhi', 10)
    run_isc('../data/byintensity/hhi_t_mli.csv', '../data/byintensity/hhi_c_full.csv', 'hh_inc_deflated', 'mli', 10)
    run_isc('../data/byintensity/hhi_t_li.csv', '../data/byintensity/hhi_c_full.csv', 'hh_inc_deflated', 'li', 10)

    run_isc('../data/byintensity/is_t_hi.csv', '../data/byintensity/is_c_full.csv', 'inc_share', 'hi', 10)
    run_isc('../data/byintensity/is_t_mhi.csv', '../data/byintensity/is_c_full.csv', 'inc_share', 'mhi', 10)
    run_isc('../data/byintensity/is_t_mli.csv', '../data/byintensity/is_c_full.csv', 'inc_share', 'mli', 10)
    run_isc('../data/byintensity/is_t_li.csv', '../data/byintensity/is_c_full.csv', 'inc_share', 'li', 10)


