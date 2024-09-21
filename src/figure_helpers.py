import os
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.nonparametric.smoothers_lowess import lowess


mpl.rcParams['font.family'] = 'Helvetica'
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(filename1, filename2):
    df_synths = pd.read_csv(filename1, index_col='year')
    df_synths = df_synths[-8:6]
    df_treats = pd.read_csv(filename2, index_col='year')
    df_treats = df_treats[-8:6]
    return [df_synths, df_treats]

def synth_treat_ax_plotter(df, ax, xlabel, ylabel, title, legend=False, n_cols=1, loc='lower left'):
    mpl.rcParams['font.family'] = 'Helvetica'
    ax.axvline(x=0, linestyle='--', color='k', alpha=1, linewidth=1)
    colors = ['#5e4fa2', '#9e0142', '#ffffbf']
    ax.plot(df[0].mean(axis=1), color=colors[0], linewidth=1.5, linestyle='-')
    ax.plot(df[1].mean(axis=1), color=colors[1], linewidth=1.5, linestyle='--')
    ax.fill_between(df[0].index,
                    y1=df[0].mean(axis=1),
                    y2=df[1].mean(axis=1),
#                    edgecolor= (25/255, 174/255, 97/2551),
#                    facecolor= (255/255, 255/255, 191/255, 0.1),
                    edgecolor=(25/255,25/255,25/255, 0.3),
                    facecolor=(220/255,220/255,220/255, 0.15),
                    linewidth=0.15,
                    hatch='..')
    ax.set_title(title, loc='left', fontsize=22)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'£{x:,.0f}'))
    ax.set_title(title, loc='left', fontsize=22)
    if 'Employed' in ylabel:
        ax.yaxis.set_major_formatter('£{x:1.0f}')
    ax.grid(linestyle='--', color='k', alpha=0.1, zorder=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(axis='both', labelsize=14, rotation=0)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    if legend:
        legend_elements = [Line2D([0], [0],
                                  linewidth=1,
                                  color=colors[0], label=r'Control'),
                           Line2D([0], [0],
                                  linewidth=1,
                                  color=colors[1], label=r'Treatment'),
                            patches.Patch(facecolor=(220/255, 220/255, 220/255, 0.2),
                                          edgecolor=(0/255, 0/255, 0/255, 0.75),
                                          hatch='..', linewidth=0.15, label='Difference')
                           ]
        ax.legend(handles=legend_elements, loc=loc, frameon=True,
                  fontsize=10, framealpha=1, facecolor='w',
                  edgecolor='k', handletextpad=0.25, ncols=n_cols
                  )


def plot_vg(path_to_data, path_to_wvar, ax):
    colors = ['#41558c', '#E89818', '#CF202A']
    differences = pd.read_csv(path_to_data, index_col=0)[-8:6]
    wvar = pd.read_csv(path_to_wvar, index_col=0)[-8:6]
    differences.mean(axis=1).plot(color=colors[0], ax=ax)
    bvar = differences.var(axis=1)
    total_var = pd.concat([wvar, bvar], axis=1).sum(axis=1)
    row_n = differences.count(axis=1)
    total_se = (total_var/row_n).apply(math.sqrt)
    ax.axvline(x=0, linestyle='dotted', color=colors[2])
    ax.axhline(y=0, linestyle='dotted', color=colors[2])
    ax.fill_between(x=differences.index,
                    y1=differences.mean(axis=1) + (1.96*total_se),
                    y2=differences.mean(axis=1) - (1.96*total_se),
                    edgecolor=(25 / 255, 25 / 255, 25 / 255, 0.3),
                    facecolor=(220 / 255, 220 / 255, 220 / 255, 0.15),
                    alpha=0.15, hatch='..')
    ax.yaxis.set_major_formatter('£{x:1.0f}')
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.tight_layout()
    sns.despine()


def bootstrap(data, num_iterations):
    boot_means = []
    for _ in range(num_iterations):
        num_samples = len(data.dropna())
        try:
            sample = np.random.choice(data.dropna(), size=num_samples, replace=True)
            boot_means.append(np.mean(sample))
        except ValueError:
            boot_means.append(np.nan)
    return boot_means


def plot_dv(path_to_data, low_lim, high_lim,
            num_iterations, ax, ylabel, title,
            xlabel='', legend=False, n_cols=1, loc='lower left'):
    differences = pd.read_csv(path_to_data, index_col=0)[-8:6]
    ci_boots = ci_bootstrap(path_to_data, low_lim, high_lim, num_iterations)[-8:6]
    neg = differences.mean(axis=1)[differences.mean(axis=1).index < 0]
    neg.plot(ax=ax, marker='o', linewidth=0, color='#5e4fa2',
             markeredgecolor='k', markeredgewidth=1, markersize=8)
    ax.bar(neg.index, ci_boots.high_ci[-8:-1] - ci_boots.low_ci[-8:-1],
           bottom=ci_boots.low_ci[-8:-1], color='#3288bd', edgecolor='k', alpha=0.7)
    pos = differences.mean(axis=1)[differences.mean(axis=1).index >= 0]
    pos.plot(ax=ax, marker='d', linewidth=0, color='#9e0142',
             markeredgecolor='k', markeredgewidth=1, markersize=8)

    ax.bar(pos.index, ci_boots.high_ci[0:] - ci_boots.low_ci[0:],
           bottom=ci_boots.low_ci[0:], color='#d53e4f', edgecolor='k', alpha=0.7)

    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_title(title, fontsize=22, loc='left')
    ax.grid(which="major", linestyle='--', alpha=0.225)
    ax.tick_params(axis='x', rotation=45)
    if 'share' in str(path_to_data):
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x:,.2f}%'))
    if 'placebo' in str(path_to_data):
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'£{x:,.2f}'))
    else:
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'£{x:,.0f}'))
    ax.tick_params(axis='both', labelsize=14, rotation=0)
    sns.despine(ax=ax)
    if legend is True:
        legend_elements1 = [Patch(facecolor='#3288bd', edgecolor='k',
                                  label='Pre-Treatment\n    CI (95%)', alpha=0.7),
                            Patch(facecolor='#d53e4f', edgecolor='k',
                                  label='Post-Treatment\n    CI (95%)', alpha=0.7),
                            Line2D([0], [0], color='#5e4fa2', lw=0, linestyle='-',
                                   markersize=10,
                                   marker='o', markeredgecolor='k', markeredgewidth=1,
                                   label='Pre-Treatment\n   Coefficient', alpha=1),
                            Line2D([0], [0], color='#9e0142', lw=0, linestyle='-',
                                   markersize=10,
                                   marker='d', markeredgecolor='k', markeredgewidth=1,
                                   label='Post-Treatment\n   Coefficient', alpha=1),
                            ]
        ax.legend(handles=legend_elements1, loc=loc, frameon=True,
                  fontsize=10, framealpha=1, facecolor='w',
                  edgecolor=(0, 0, 0, 1), ncols=n_cols
                  )
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)


def ci_bootstrap(path_to_data, low_lim, high_lim, num_iterations=1000):
    df = pd.read_csv(path_to_data, index_col=0)
    bootstrap_results = {row: bootstrap(df.loc[row], num_iterations) for row in df.index}
    quantiles = {row: (np.percentile(bootstrap_results[row], low_lim),
                       np.percentile(bootstrap_results[row], high_lim)) for row in df.index}
    quantiles_df = pd.DataFrame(quantiles, index=['low_ci', 'high_ci']).T
    return quantiles_df


def plot_EDA_figure(result, df):
    fig = plt.figure(figsize=(14, 11.5), constrained_layout=True)
    gs = gridspec.GridSpec(8, 4, figure=fig)

    ax1 = fig.add_subplot(gs[:4, 0:2])
    ax2 = fig.add_subplot(gs[:4, 2:4])
    ax3 = fig.add_subplot(gs[4:, 0:2])
    ax4 = fig.add_subplot(gs[4:6, 2:4])
    ax5 = fig.add_subplot(gs[6:8, 2:4])

    colors = ['#2b83ba', '#fdae61', '#abdda4', '#d7191c']

    # Figure a.
    result.plot(kind='bar', stacked=True, edgecolor='k', ax=ax1, color=colors)
    legend_elements1 = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'0-4 Hours'),
        Patch(facecolor=colors[1], edgecolor=(0, 0, 0, 1),
              label=r'5-10 Hours'),
        Patch(facecolor=colors[2], edgecolor=(0, 0, 0, 1),
              label=r'10-19 Hours'),
        Patch(facecolor=colors[3], edgecolor=(0, 0, 0, 1),
              label=r'20-49 Hours'),
    ]
    legend = ax1.legend(handles=legend_elements1, loc='upper right', frameon=True,
                        fontsize=11, framealpha=1, facecolor='w',
                        edgecolor=(0, 0, 0, 1), ncols=1,
                        title='   Care\nIntensity'
                        )
    plt.setp(legend.get_title(), fontsize=13)

    # Figure b.
    hb = ax2.hexbin(df['hh_inc_deflated'], df['ind_inc_deflated'], cmap='Spectral_r', gridsize=25,
                    mincnt=1, linewidths=0.15, edgecolor='k', norm=LogNorm())
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(hb, cax=cax, orientation='vertical')
    cbar.set_label('Number of Observations', fontsize=14)  # Optional: label for colorbar

    grouped_low = pd.DataFrame(df[(df['care_intensity_t'] == '0-4 hours') & (df['sex'] == 'female')].groupby(['dvage'])[
                                   'ind_inc_deflated'].mean()).reset_index()
    grouped_medlow = pd.DataFrame(
        df[(df['care_intensity_t'] == '5-19 hours') & (df['sex'] == 'female')].groupby(['dvage'])[
            'ind_inc_deflated'].mean()).reset_index()
    grouped_medhigh = pd.DataFrame(
        df[(df['care_intensity_t'] == '10-19 hours') & (df['sex'] == 'female')].groupby(['dvage'])[
            'ind_inc_deflated'].mean()).reset_index()
    grouped_high = pd.DataFrame(
        df[(df['care_intensity_t'] == '20-49 hours') & (df['sex'] == 'female')].groupby(['dvage'])[
            'ind_inc_deflated'].mean()).reset_index()
    lowess_smoothed_low = lowess(grouped_low['ind_inc_deflated'], grouped_low['dvage'], frac=0.3)
    lowess_smoothed_medlow = lowess(grouped_medlow['ind_inc_deflated'], grouped_medlow['dvage'], frac=0.3)
    lowess_smoothed_medhigh = lowess(grouped_medhigh['ind_inc_deflated'], grouped_medhigh['dvage'], frac=0.3)
    lowess_smoothed_high = lowess(grouped_high['ind_inc_deflated'], grouped_high['dvage'], frac=0.3)
    smoothed_df = pd.DataFrame(lowess_smoothed_low, columns=['dvage', 'ind_inc_deflated_smoothed'])
    smoothed_df.set_index('dvage').plot(ax=ax3, color=colors[0], legend=False)
    smoothed_df = pd.DataFrame(lowess_smoothed_medlow, columns=['dvage', 'ind_inc_deflated_smoothed'])
    smoothed_df.set_index('dvage').plot(ax=ax3, color=colors[1], legend=False)
    smoothed_df = pd.DataFrame(lowess_smoothed_medhigh, columns=['dvage', 'ind_inc_deflated_smoothed'])
    smoothed_df.set_index('dvage').plot(ax=ax3, color=colors[2], legend=False)
    smoothed_df = pd.DataFrame(lowess_smoothed_high, columns=['dvage', 'ind_inc_deflated_smoothed'])
    smoothed_df.set_index('dvage').plot(ax=ax3, color=colors[3], legend=False)

    legend_elements3 = [
        Line2D([0], [0], color=colors[0], linestyle='-',
               label='0-4 Hours', lw=1.75),
        Line2D([0], [0], color=colors[1], linestyle='-',
               label='5-9 Hours', lw=1.75),
        Line2D([0], [0], color=colors[2], linestyle='-',
               label='10-19 Hours', lw=1.75),
        Line2D([0], [0], color=colors[3], linestyle='-',
               label='20-49 Hours', lw=1.75),
    ]
    legend = ax3.legend(handles=legend_elements3, loc='upper right', frameon=True,
                        fontsize=11, framealpha=1, facecolor='w',
                        edgecolor=(0, 0, 0, 1), ncols=1,
                        title='   Care\nIntensity'
                        )
    plt.setp(legend.get_title(), fontsize=13)
    grouped_df = df.groupby(['care_intensity_t', 'sex']).size().unstack()
    grouped_df = grouped_df.reindex(['0-4 hours', '5-19 hours', '10-19 hours', '20-49 hours'])
    index = np.arange(len(grouped_df))
    bar_width = 0.35
    ax4.bar(index - bar_width / 2, grouped_df['female'], bar_width, label='Female', color=colors[0], edgecolor='k')
    ax4.bar(index + bar_width / 2, grouped_df['male'], bar_width, label='Male', color=colors[3], edgecolor='k')
    ax4.set_xticks(index)
    ax4.set_xticklabels(['0-4 Hours', '5-9 Hours', '10-19 Hours', '20-49 Hours'])
    legend_elements4 = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'Female'),
        Patch(facecolor=colors[3], edgecolor=(0, 0, 0, 1),
              label=r'Male'),
    ]
    ax4.legend(handles=legend_elements4, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1,
               )

    df['ethnicity_2'] = np.where(df['ethn_5'] == 'white', 'White', 'Non-White')
    grouped_df = df.groupby(['care_intensity_t', 'ethnicity_2']).size().unstack()
    grouped_df = grouped_df.reindex(['0-4 hours', '5-19 hours', '10-19 hours', '20-49 hours'])
    index = np.arange(len(grouped_df))
    bar_width = 0.35
    ax5.bar(index - bar_width / 2, grouped_df['White'], bar_width, label='Female', color=colors[0], edgecolor='k')
    ax5.bar(index + bar_width / 2, grouped_df['Non-White'], bar_width, label='Male', color=colors[3], edgecolor='k')
    ax5.set_xticks(index)
    ax5.set_xticklabels(['0-4 Hours', '5-9 Hours', '10-19 Hours', '20-49 Hours'])
    legend_elements5 = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'White'),
        Patch(facecolor=colors[3], edgecolor=(0, 0, 0, 1),
              label=r'Non-White'),
    ]
    ax5.legend(handles=legend_elements5, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1,
               )

    for ax, title in zip([ax1, ax2, ax3, ax4, ax5], ['a.', 'b.', 'c.', 'd.', 'e.']):
        ax.tick_params(width=1, length=8, axis='both', which='major', labelsize=15)
        ax.set_title(title, loc='left', fontsize=22, y=1.025, x=-.05)
        ax.grid(which="major", linestyle='--', alpha=0.225)

    ax1.set_xlabel('Treated Cases by Intensity', fontsize=16)
    ax2.set_xlabel('Age', fontsize=16)
    ax3.set_xlabel('Age', fontsize=16)

    ax1.set_ylabel('Frequency', fontsize=16)
    ax4.set_ylabel('Frequency', fontsize=16)
    ax5.set_ylabel('Frequency', fontsize=16)
    ax2.set_ylabel('Individual Monthly Income', fontsize=16)
    ax3.set_ylabel('Individual Monthly Income', fontsize=16)
    ax2.set_xlabel('Household Monthly Income', fontsize=16)
    ax1.set_xticklabels(result.index, rotation=0)

    def pound_formatter(x, pos):
        return '£{:,.1f}k'.format(x / 1000)

    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(pound_formatter))
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x / 1000:,.0f}k'))
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'£{x / 1000:,.0f}k'))
    ax2.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'£{x / 1000:,.0f}k'))
    ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x / 1000:,.0f}k'))
    ax5.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x / 1000:,.0f}k'))
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(5))
    sns.despine()
    plt.savefig(os.path.join('..',
                             'figs',
                             'eda_figure.pdf'
                             ),
                bbox_inches='tight')


def make_weighted_intensity_table(df):
    filtered_df = df[(df['reindex'] >= -8) & (df['reindex'] <= 8)]
    weighted_crosstab = pd.crosstab(
        index=filtered_df['reindex'],
        columns=filtered_df['care_intensity_t'],
        values=filtered_df['weight_yearx'],
        aggfunc='sum',
        dropna=True
    ).fillna(0)
    result = pd.concat([weighted_crosstab], keys=['Freq'])
    result = result.reset_index(level=[0, 1]).drop('level_0', axis=1).set_index('reindex')
    result = result[['0-4 hours', '5-19 hours', '10-19 hours', '20-49 hours']]
    result.to_csv(os.path.join(os.getcwd(), '..', 'tables', 'weighted_frequency.csv'))
    print(result)
    return result


def axis_sdid_plotter(df, ax, ylabel, title, xlabel='', legend=False):
    ax.axvline(x=5, color='k', linestyle='--', linewidth=1)
    ax.plot(df['Control'], color='#5e4fa2', marker='o', markersize=12,
            markerfacecolor='w', markeredgecolor='k')
    ax.plot(df['Treatment '], color='#9e0142', marker='o', markersize=12,
            markerfacecolor='w', markeredgecolor='k')
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.grid(which="major", linestyle='--', alpha=0.225)
    ax.set_title(title, fontsize=22, loc='left')
    ax.tick_params(axis='x', rotation=45)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'£{x:,.0f}'))
    ax.tick_params(axis='both', labelsize=14, rotation=0)
    ax.set_xticklabels(range(-5, 6));
    sns.despine(ax=ax)
    if legend is True:
        legend_elements1 = [Line2D([0], [0], color='#5e4fa2', lw=1, linestyle='-',
                                   markersize=10, markerfacecolor='w',
                                   marker='o', markeredgecolor='k', markeredgewidth=1,
                                   label='Control\n Group', alpha=1),
                            Line2D([0], [0], color='#9e0142', lw=1, linestyle='-',
                                   markersize=10, markeredgecolor='k', markerfacecolor='w',
                                   marker='o', markeredgewidth=1,
                                   label='Treatment\n  Group', alpha=1),
                           ]
        ax.legend(handles=legend_elements1, loc='upper right', frameon=True,
                  fontsize=10, framealpha=1, facecolor='w',
                  edgecolor=(0, 0, 0, 1), ncols=2
                  )
    return ax


def axis_did_plotter(df, ax, ylabel, title, xlabel='', legend=False):
    df['Coefficient'][0:8].plot(ax=ax, marker='o', linewidth=0, color='#5e4fa2',
                                markeredgecolor='k', markeredgewidth=1, markersize=8)
    ax.bar(df['Unnamed: 0'][0:8], df['[95% conf.'][0:8] - df['interval]'][0:8],
           bottom=df['interval]'][0:8], color='#3288bd', edgecolor='k', alpha=0.7)
    df['Coefficient'][8:].plot(ax=ax, marker='d', linewidth=0, color='#9e0142',
                               markeredgecolor='k', markeredgewidth=1, markersize=8)
    ax.bar(df['Unnamed: 0'][8:], df['[95% conf.'][8:] - df['interval]'][8:],
           bottom=df['interval]'][8:], color='#d53e4f', edgecolor='k', alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_title(title, fontsize=22, loc='left')
    ax.grid(which="major", linestyle='--', alpha=0.225)
    ax.tick_params(axis='x', rotation=45)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'£{x:,.0f}'))
    ax.tick_params(axis='both', labelsize=14, rotation=0)
    sns.despine(ax=ax)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    if legend is True:
        legend_elements1 = [Patch(facecolor='#3288bd', edgecolor='k',
                            label='Pre-Treatment\n    CI (95%)', alpha=0.7),
                           Patch(facecolor='#d53e4f', edgecolor='k',
                            label='Post-Treatment\n    CI (95%)', alpha=0.7),
                            Line2D([0], [0], color='#5e4fa2', lw=0, linestyle='-',
                                   markersize=10,
                                   marker='o', markeredgecolor='k', markeredgewidth=1,
                                   label='Pre-Treatment\n   Coefficient', alpha=1),
                            Line2D([0], [0], color='#9e0142', lw=0, linestyle='-',
                                   markersize=10,
                                   marker='d', markeredgecolor='k', markeredgewidth=1,
                                   label='Post-Treatment\n   Coefficient', alpha=1),
                           ]
        ax.legend(handles=legend_elements1, loc='upper right', frameon=True,
                  fontsize=10, framealpha=1, facecolor='w',
                  edgecolor=(0, 0, 0, 1), ncols=2
                  )
    ax.set_xticklabels(range(-8, 7));
    return ax