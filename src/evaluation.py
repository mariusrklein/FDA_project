from typing import Dict, Tuple, Callable
import os
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy.builtins import Q
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse import csc_matrix, csr_matrix
from itertools import chain
import anndata as ad
from src.correction import normalize_proportion_ratios
from src import const
import seaborn as sns
import scanpy as sc
import re



## ------------------------ ##
##   EVALUATION FUNCTIONS   ##
## ------------------------ ##


def plot_all_ion_slopes(
    am_adata, 
    subset = [], 
    col_wrap = 10, 
    ratios = True,
    log = True,
    kde = False):

    norm_int = am_adata.copy()
    if ratios:
        norm_int = normalize_proportion_ratios(am_adata, normalized=True)


    ions = [] 
    if len(subset) == 0:
        ions = list(am_adata.var_names)
    else:
        ions = list(subset)
    
    ions.append(const.TPO)
    # print(ions)
    grid_slopes_df = sc.get.obs_df(norm_int, keys=ions)
    grid_slopes_df['am'] = grid_slopes_df.index
    plot_df = grid_slopes_df.melt(id_vars=['am', const.TPO], var_name='ion')
    if log:
        plot_df[[const.TPO, 'value']] = np.log10(plot_df[[const.TPO, 'value']])
    grid = sns.FacetGrid(plot_df, col = 'ion', col_wrap = col_wrap)
    if kde:
        grid.map(sns.kdeplot, const.TPO, 'value', fill=True)
    else:
        grid.map(sns.scatterplot, const.TPO, 'value')
    for i, ax in enumerate(grid.axes.flat): 
        ax.axline((0,0), slope=-1)
        ax.set(ylabel='log intensity / sampling prop. ratio', xlabel = 'log sampling proportion')


def analyse_corrected_metabolites(adata,
    adata_cor,
    condition_name,
    top_ions = 5,
    save_to_path = None,
    exclude_pool_corrected = True,
    volcano_plot = True,
    pair_plot = True,
    top_plot = True
):
    conc_adata = ad.concat({'uncorrected': adata, 'ISM correction': adata_cor}, label='correction', index_unique='_', merge='same')
    
    sc.tl.rank_genes_groups(adata=conc_adata, groupby='correction', use_raw=True, method='wilcoxon')
    all_ions_dea = sc.get.rank_genes_groups_df(conc_adata, group='ISM correction')

    adata_cor.var['names'] = adata_cor.var.index
    impact_ions = pd.merge(all_ions_dea, adata_cor.var[['names', 'correction_using_ion_pool', 'correction_n_iterations', 'correction_n_datapoints', 'mean_correction_quantreg_slope']], on='names', how='left')
    impact_ions = impact_ions.sort_values(by='scores')
    impact_ions['logfoldchanges'] = impact_ions['logfoldchanges'].replace(-np.Inf, min(impact_ions['logfoldchanges'])*1.1)

    impact_ions['significant'] = (impact_ions['pvals'] < 0.001) & (impact_ions['scores'] < -2)
    sign_impact_ions = impact_ions[impact_ions['significant'] == True]
    print('Of %1d ions, %1d are significantly altered by ion suppression correction (p < 1e-3, DEA score < -2).'%(len(impact_ions), len(sign_impact_ions)))
    
    print('Association of significance and pool correction')
    print(pd.crosstab(impact_ions['significant'], impact_ions['correction_using_ion_pool'], margins=True))
    included_molecules = list(impact_ions['names'])
    if exclude_pool_corrected:
        print('excluding pool_corrected')
        impact_ions = impact_ions[impact_ions['correction_using_ion_pool'] == False]
        included_molecules = list(impact_ions.loc[impact_ions['correction_using_ion_pool'] == False, 'names'])
    
    if volcano_plot:
        print('volcano plot of changed intensities')
        volcano_df = impact_ions
        volcano_df['-log pval'] = -np.log10(volcano_df['pvals'])
        fig, ax = plt.subplots(1,2)
        sns.scatterplot(volcano_df, x='logfoldchanges', y='-log pval', ax=ax[0], hue='significant')
        sns.scatterplot(volcano_df, x='scores', y='-log pval', ax=ax[1], hue='significant')
        ax[0].set_title('Intensity LFC by ion suppression correction')
        ax[1].set_title('Corresponding scores')
        fig.set_figwidth(12)
    
    conc_adata_raw = conc_adata.copy()
    conc_adata_raw.obs['cell'] = [re.sub('_[a-zA-Z ]+$', '', i) for i in conc_adata_raw.obs.index]
    conc_adata_raw.X = conc_adata_raw.raw.X
    changed_ions_df = sc.get.obs_df(conc_adata_raw, keys=(['correction', condition_name, 'cell', 'well']+list(conc_adata.var_names)))
    plot_df = changed_ions_df.melt(id_vars=['correction', condition_name, 'cell', 'well'], var_name='ion').pivot(index=['ion', condition_name, 'cell', 'well'], columns='correction', values='value')
    plot_df.reset_index(inplace=True)
    plot_df = plot_df[plot_df.uncorrected > 0]
    
    if pair_plot:
        def correlate_ions(ion):
            return plot_df[plot_df.ion == ion].corr(numeric_only = True)['uncorrected']['ISM correction']
        print('associations between measures')
        impact_ions['pearson'] = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(correlate_ions)(ion) for ion in tqdm(impact_ions['names']))
        sns.pairplot(impact_ions[['scores', 'logfoldchanges', 'mean_correction_quantreg_slope', 'pearson', 
                                  'significant']], 
                     hue='significant'
                    )
        if exclude_pool_corrected == False:
            sns.pairplot(impact_ions[['scores', 'logfoldchanges', 'mean_correction_quantreg_slope', 'pearson', 
                                      'mean_correction_quantreg_slope']], 
                         hue='mean_correction_quantreg_slope'
                        )
     
    if top_plot:
        print('top most and least corrected ions')
        ions_corr = list(impact_ions['names'].head(top_ions))
        ions_uncorr = list(impact_ions['names'].tail(top_ions))
        ions_max_slope = list(impact_ions.sort_values(by='mean_correction_quantreg_slope')['names'].head(top_ions))
        ions_min_slope = list(impact_ions.sort_values(by='mean_correction_quantreg_slope')['names'].tail(top_ions))
        ions = ions_corr + ions_uncorr + ions_max_slope + ions_min_slope
        # simple way to remove duplicates
        ions = list(dict.fromkeys(ions))
        plot_df['group'] = ['strongly corr' if (ion in ions_corr) 
                            else 'hardly corr' if (ion in ions_uncorr)
                            else 'hardly corr' if (ion in ions_uncorr)
                            else 'max slope' if (ion in ions_max_slope)
                            else 'min_slope' if (ion in ions_min_slope)
                            else 'other' for ion in plot_df['ion'] ]

        slopes = [adata_cor.var.loc[ion, 'correction_quantreg_slope'] for ion in ions]
        # intersects = [adata_cor.var.loc[ion, 'correction_quantreg_intersect'] for ion in ions]
        pearson = [plot_df[plot_df.ion == ion].corr(numeric_only=True)['uncorrected']['ISM correction'] for ion in ions]
        scores = [float(impact_ions.loc[impact_ions.names == ion, 'scores']) for ion in ions]
        grid = sns.FacetGrid(plot_df, col='ion', hue='group', col_wrap=5, sharex=False, sharey=False, col_order=ions, palette='cividis')
        grid.map(sns.scatterplot, 'uncorrected', 'ISM correction').add_legend()
        grid.set(aspect = 1)
        for i, ax in enumerate(grid.axes.flat): 
            lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
            lim_min = min([ax.get_xlim()[1], ax.get_ylim()[1]])
            textstr = 'quantreg slope = %1.3f\npearson r = %1.5f\nscores = %1.2f'%(slopes[i], pearson[i], scores[i])
            props = dict(boxstyle='round', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props, fontsize=10)
            ax.axline((lim_min,lim_min), (lim,lim))
            
    if save_to_path is not None:
        subset_adata = conc_adata_raw.copy()
        df = subset_adata.to_df()
        # compress data by rounding
        significant_figures = int(df.shape[0] * df.shape[1] / 6e6)
        df = df.applymap(lambda x: 0 if x == 0 else round(x, significant_figures - int(np.floor(np.log10(np.abs(x))))))
        df.to_csv(os.path.join(save_to_path, 'metenrichr_matrix.csv'))

        metadata = pd.DataFrame(subset_adata.obs['correction'])
        metadata['correction'].name = 'condition'
        metadata.to_csv(os.path.join(save_to_path, 'metenrichr_metadata.csv'))
    
    return (impact_ions, plot_df)
        
    


def plot_intermixing_graph(ax, summary_df, label):
    ax.plot(summary_df.index, summary_df['mean'], label=label)
    # ax.fill_between(summary_df.index, (summary_df['mean'] - summary_df['sd']), summary_df['mean'] + summary_df['sd'], alpha=0.2)
    # ax.set_ylim(0, 1.05)
    # ax.set_xscale('log')
    ax.set_xlabel('Relative neighborhood size')
    ax.set_ylabel('Cluster intermixing')

def intermixing_metric_sampled(
    adata_orig: ad.AnnData,
    condition_name: str,
    sample_frac=0.2,
    ax=None,
    label="",
    measure = 'X_umap',
    n_datapoints = 100,
    neighborhood_size = 100,
    normalized = False,
    n_jobs = multiprocessing.cpu_count()
):

    sample = adata_orig.obs.groupby(condition_name,
        group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=1))

    adata = adata_orig.copy()
    adata = adata[sample.index]

    dist_matrix: np.ndarray
    if measure == "X":
        dist_matrix = distance_matrix(adata.X,
            adata_orig.X)
    else:
        dist_matrix = distance_matrix(adata.obsm[measure],
            adata_orig.obsm[measure])
    #neighborhood_size = len(adata.obs)
    #sampling_range = np.unique(np.logspace(0, np.log10(neighborhood_size), n_datapoints).astype(int))
    
    if n_datapoints == 1:
        sampling_range = list(chain(*[neighborhood_size]))
    else:
        sampling_range = range(1, neighborhood_size, 
            round(neighborhood_size / n_datapoints))
    
    norm_factors = ( adata.obs[condition_name].value_counts() / len(adata.obs) 
        * len(adata.obs[condition_name].value_counts()) )

    #neighborhood_df = pd.DataFrame(columns=sampling_range, index=adata.obs['ObjectNumber'])

    def get_neighborhood_series(index, celltype):
        neighbors = pd.Series(dist_matrix[index],
            index=adata_orig.obs[condition_name]).sort_values()
        if normalized:
            return [ 1 - (neighbors[:i].index.value_counts()[celltype] / i / 
                norm_factors[celltype])  for i in sampling_range]
        return [ 1 - (neighbors[:i].index.value_counts()[celltype] / i) for i in sampling_range]

    neighborhood_df = pd.DataFrame(
        Parallel(n_jobs=n_jobs)(delayed(get_neighborhood_series)(index, celltype) 
        for index, celltype in enumerate(adata.obs[condition_name])),
        columns=sampling_range, index=adata.obs.index
    )
    
    summary = pd.concat([neighborhood_df.mean(axis=0), neighborhood_df.std(axis=0)], axis=1)
    summary.columns = ["mean", "sd"]
    summary['rel_neighborhood'] = np.linspace(0, 1, len(summary))
    if ax is not None:
        plot_intermixing_graph(ax, summary, label)

    return summary


def intermixing(adata_dict: Dict[str, ad.AnnData],
    condition_name: str,
    measures = ['X', 'X_pca', 'X_umap'],
    show_table = [10],
    sample_frac = 0.2,
    n_datapoints = 100,
    neighborhood_size = 100,
    n_jobs = multiprocessing.cpu_count()
)  -> pd.DataFrame:

    fig, ax = plt.subplots(1, len(measures), sharey=True)
    if len(measures) == 1:
        ax = [ax]
        
    fig.set_figwidth(4*len(measures))
    
    summaries = {}

    for i, measure in enumerate(measures):
        print(measure)
        for label, adata in adata_dict.items():
            summaries[measure+'_'+label] = intermixing_metric_sampled(adata_orig = adata, 
                condition_name = condition_name, 
                sample_frac = sample_frac, 
                ax = ax[i], 
                label = measure+'_'+label, 
                measure = measure,
                n_datapoints = n_datapoints,
                neighborhood_size = neighborhood_size,
                normalized = True,
                n_jobs = n_jobs
            )

        ax[i].legend()



    fig.tight_layout()
    
    if not all([hood in list(summaries.values())[0].index for hood in show_table]):
        summaries = {}
        for i, measure in enumerate(measures):
            for label, adata in adata_dict.items():
                summaries[measure+'_'+label] = intermixing_metric_sampled(adata_orig = adata, 
                    condition_name = condition_name,
                    neighborhood_size = show_table,
                    sample_frac = sample_frac,
                    label = measure+'_'+label,
                    measure = measure,
                    normalized = True,
                    n_jobs = n_jobs
                )

    return pd.concat({k: v.loc[show_table] for k, v in summaries.items()})
   