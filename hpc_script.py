import pandas as pd
from utils import get_control_clean
from utils import isc


def main():
    treated = pd.read_csv('treated.csv', index_col=0)
    controls = pd.read_csv('control.csv', index_col=0)
    controls = controls[~(controls.pidp == 1020477375)]
    print('Data loaded...')
    samples_fc = get_control_clean(controls, treated,
                                ['inc_share',
                                 'dvage',
                                 'sex_recoded',
                                 'mastat_recoded',
                                 'asian',
                                 'black',
                                 'mixed',
                                 'other',
                                 'low',
                                 'middle'],
                                'inc_share')
    samples_dc = get_control_clean(controls, treated, ['inc_share', 'dvage', 'sex_recoded', 'asian', 'black', 'mixed', 'other'], 'inc_share')
    samples_nc = get_control_clean(controls, treated, ['inc_share'], 'inc_share')
    print('Data cleaned...')
    out_nc = isc(samples_nc)
    out_dc = isc(samples_dc)
    out_fc = isc(samples_fc)
    
    diffs_nc = pd.concat(out_nc['diffs'], axis=1).sort_index()
    treats_nc = pd.concat(out_nc['treats'], axis=1).sort_index()
    synths_nc = pd.concat(out_nc['synths'], axis=1).sort_index()
    
    diffs_dc = pd.concat(out_dc['diffs'], axis=1).sort_index()
    treats_dc = pd.concat(out_dc['treats'], axis=1).sort_index()
    synths_dc = pd.concat(out_dc['synths'], axis=1).sort_index()

    diffs_fc = pd.concat(out_fc['diffs'], axis=1).sort_index()
    treats_fc = pd.concat(out_fc['treats'], axis=1).sort_index()
    synths_fc = pd.concat(out_fc['synths'], axis=1).sort_index()

    print('Synth controls created...')
    diffs_nc.to_csv('diffs_nc_emp.csv')
    treats_nc.to_csv('treats_nc_emp.csv')
    synths_nc.to_csv('synths_nc_emp.csv')
    
    diffs_dc.to_csv('diffs_dc_emp.csv')
    treats_dc.to_csv('treats_dc_emp.csv')
    synths_dc.to_csv('synths_dc_emp.csv')

    diffs_fc.to_csv('diffs_fc_emp.csv')
    treats_fc.to_csv('treats_fc_emp.csv')
    synths_fc.to_csv('synths_fc_emp.csv')

    print('Synth controls saved...')


if __name__ == "__main__":
    main()
