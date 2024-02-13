import pandas as pd
from utils import isc_b
import pickle


def run_isc_short(samples_pkl, target_var, out_suffix, k_n=35):
    with open(samples_pkl, 'rb') as file:
        samples = pickle.load(file)
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
    run_isc_short('./outputs/ind_inc_deflated_hi.pkl', 'ind_inc_deflated', 'hi')
    run_isc_short('./outputs/ind_inc_deflated_li.pkl', 'ind_inc_deflated', 'li', k_n=500)


if __name__ == "__main__":
    main()
