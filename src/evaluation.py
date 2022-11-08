from typing import Dict, Tuple, Callable
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



## ------------------------ ##
##   EVALUATION FUNCTIONS   ##
## ------------------------ ##


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
        for index, celltype in tqdm(enumerate(adata.obs[condition_name]))),
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
    n_jobs = multiprocessing.cpu_count()
)  -> pd.DataFrame:

    fig, ax = plt.subplots(1, len(measures), sharey=True)
    fig.set_figwidth(4*len(measures))
    
    summaries = {}

    for i, measure in enumerate(measures):
        
        for label, adata in adata_dict.items():
            summaries[measure+'_'+label] = intermixing_metric_sampled(adata_orig = adata, 
                condition_name = condition_name, 
                sample_frac = sample_frac, 
                ax = ax[i], 
                label = measure+'_'+label, 
                measure = measure,
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
   