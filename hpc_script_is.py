import pandas as pd
from utils import get_control_clean
from utils import isc


def main():
    treated = pd.read_csv('./data/is_t_full.csv', index_col=0)
    controls = pd.read_csv('./data/is_c_full.csv', index_col=0)
    controls = controls[~(controls.pidp == 1020477375)]
    target_var = 'inc_share'
    print('Data loaded...')
    samples_fc = get_control_clean(controls, treated,
                                [target_var,
                                 'dvage',
                                 'sex_recoded',
                                 'mastat_recoded',
                                 'asian',
                                 'black',
                                 'mixed',
                                 'other',
                                 'low',
                                 'middle'],
                                   target_var,
                                 'weight_yearx')
    samples_dc = get_control_clean(controls, treated, [target_var, 'dvage', 'sex_recoded', 'asian', 'black', 'mixed', 'other'], target_var, 'weight_yearx')
    samples_nc = get_control_clean(controls, treated, [target_var], target_var, 'weight_yearx')
    print('Data cleaned...')
    out_nc = isc(samples_nc)
    out_dc = isc(samples_dc)
    out_fc = isc(samples_fc)
    
    diffs_nc = pd.concat(out_nc['diffs'], axis=1).sort_index()
    w_diffs_nc = pd.concat(out_nc['weighted_diff'], axis=1).sort_index()
    treats_nc = pd.concat(out_nc['treats'], axis=1).sort_index()
    synths_nc = pd.concat(out_nc['synths'], axis=1).sort_index()
    
    diffs_dc = pd.concat(out_dc['diffs'], axis=1).sort_index()
    w_diffs_dc = pd.concat(out_dc['weighted_diff'], axis=1).sort_index()
    treats_dc = pd.concat(out_dc['treats'], axis=1).sort_index()
    synths_dc = pd.concat(out_dc['synths'], axis=1).sort_index()

    diffs_fc = pd.concat(out_fc['diffs'], axis=1).sort_index()
    w_diffs_fc = pd.concat(out_fc['weighted_diff'], axis=1).sort_index()
    treats_fc = pd.concat(out_fc['treats'], axis=1).sort_index()
    synths_fc = pd.concat(out_fc['synths'], axis=1).sort_index()

    print('Synth controls created...')
    diffs_nc.to_csv(f'diffs_nc_{target_var}.csv')
    w_diffs_nc.to_csv(f'w_diffs_nc_{target_var}.csv')
    treats_nc.to_csv(f'treats_nc_{target_var}.csv')
    synths_nc.to_csv(f'synths_nc_{target_var}.csv')
    
    diffs_dc.to_csv(f'diffs_dc_{target_var}.csv')
    w_diffs_dc.to_csv(f'w_diffs_dc_{target_var}.csv')
    treats_dc.to_csv(f'treats_dc_{target_var}.csv')
    synths_dc.to_csv(f'synths_dc_{target_var}.csv')

    diffs_fc.to_csv(f'diffs_fc_{target_var}.csv')
    w_diffs_fc.to_csv(f'w_diffs_fc_{target_var}.csv')
    treats_fc.to_csv(f'treats_fc_{target_var}.csv')
    synths_fc.to_csv(f'synths_fc_{target_var}.csv')

    print('Synth controls saved...')


if __name__ == "__main__":
    main()
