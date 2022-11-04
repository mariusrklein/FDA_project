"""Functions for the calculation of corrected ion intensity data
to mitigate effects of ion suppression

Functions:

Author: Marius Klein (mklein@duck.com), October 2022
"""
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
from itertools import chain
import anndata as ad



# CONSTANTS

CELL_PRE = ''
PIXEL_PRE = ''


def get_matrices(
    mark_area: Dict[str, list],
    marks_cell_associations: Dict[str, list],
    marks_cell_overlap: Dict[str, int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """calculates overlap_matrix and sampling_spec_matrix
    for given positional pixel/cell data

    Args:
        mark_area (Dict[str, list]): 
        marks_cell_associations (Dict[str, list]): [description]
        marks_cell_overlap (Dict[str, int]): [description]

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: [description]
    """

    # prototype pixel x cell matrix for abs. area overlap of all possible pixel-cell combinations
    overlap_matrix = pd.DataFrame(index=[CELL_PRE + n for n in marks_cell_overlap.keys()],
                                             columns=[PIXEL_PRE + n for n in mark_area.keys()])

    # analogous matrix for overlap relative to each pixel area
    # (corresponds to ablated region specific sampling proportion)
    sampling_prop_matrix = overlap_matrix.copy()

    # analogous matrix for constitution of every cell
    sampling_spec_matrix = overlap_matrix.copy()


    for cell_i in marks_cell_associations.keys():
        pixels = marks_cell_associations[cell_i]
        # get the index of the pixels as their order is the same in marks_cell_overlap
        for pixel_i, pixel_loc in enumerate(pixels):

            overlap_area = marks_cell_overlap[cell_i][pixel_i]
            # write absolute area overlap of current cell-pixel association to respective
            # location in matrix
            overlap_matrix.loc[CELL_PRE + cell_i, PIXEL_PRE + pixel_loc] = overlap_area

    total_pixel_size = dict(zip(overlap_matrix.columns, mark_area.values()))
    sampling_prop_matrix = overlap_matrix.divide(total_pixel_size, axis=1).replace(np.nan, 0)

    total_cell_coverage = overlap_matrix.sum(axis=1).replace(to_replace=0, value=1)
    sampling_spec_matrix = overlap_matrix.divide(total_cell_coverage, axis=0).replace(np.nan, 0)

    return(sampling_prop_matrix, sampling_spec_matrix)



def get_matrices_from_dfs(
    mark_area: pd.DataFrame,
    cell_area: pd.DataFrame,
    marks_cell_overlap: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """calculates overlap_matrix and sampling_spec_matrix
    for given positional pixel/cell data

    Args:
        mark_area (pd.DataFrame): Data frame of all ablation marks. Has identifier column `am_id` 
        cell_area (pd.DataFrame): Data frame of all captured cells. Has identifier column `cell_id`
        marks_cell_overlap (pd.DataFrame): Data frame of all overlaps between ablation marks and
        cells. Has identifier columns `cell_id` and `am_id` and an `area` column that contains the
        area of a corresponding overlap.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Returns two dataframes, the normalized overlap matrix and
        the sampling-specificity matrix.
    """

    # prototype pixel x cell matrix for abs. area overlap of all possible pixel-cell combinations
    overlap_matrix = pd.DataFrame(index=[CELL_PRE + str(n) for n in cell_area.cell_id.astype(int)],
                                             columns=[PIXEL_PRE + str(n) for n in mark_area.am_id.astype(int)])

    # analogous matrix for overlap relative to each pixel area
    # (corresponds to ablated region specific sampling proportion)
    sampling_prop_matrix = overlap_matrix.copy()

    # analogous matrix for constitution of every cell
    sampling_spec_matrix = overlap_matrix.copy()


    for cell_i in cell_area.cell_id:
        pixels = marks_cell_overlap[marks_cell_overlap.cell_id == cell_i]
        # get the index of the pixels as their order is the same in marks_cell_overlap
        for _, pixel_row in pixels.iterrows():
            # write absolute area overlap of current cell-pixel association to respective
            # location in matrix
            overlap_matrix.loc[CELL_PRE + str(int(cell_i)), PIXEL_PRE + str(int(pixel_row['am_id']))] = pixel_row['area']

    total_pixel_size = pd.Series(mark_area.area).replace(0, np.nan)
    total_pixel_size.index = overlap_matrix.columns

    sampling_prop_matrix = overlap_matrix.divide(total_pixel_size, axis=1).replace(np.nan, 0)

    total_cell_coverage = overlap_matrix.sum(axis=1).replace(to_replace=0, value=1)
    sampling_spec_matrix = overlap_matrix.divide(total_cell_coverage, axis=0).replace(np.nan, 0)

    return(sampling_prop_matrix, sampling_spec_matrix)

def add_matrices(adata: ad.AnnData, 
    overlap_matrix: pd.DataFrame, 
    sampling_spec_matrix: pd.DataFrame
    ) -> None:
    """attaches pixel-cell matrices to an AnnData object for downstream calculations. The referenced
    AnnData object is manipulated inplace.

    Args:
        adata (ad.AnnData): pixel-based AnnData object
        overlap_matrix (pd.DataFrame): A normalised overlap-matrix with cells in rows and pixels in columns
        sampling_spec_matrix (pd.DataFrame): Analoous to overlap-matrix: a Sampling specificity matrix.
    """

    adata.obsm['correction_overlap_matrix'] = overlap_matrix.T.to_numpy()
    adata.obsm['correction_sampling_spec_matrix'] = sampling_spec_matrix.T.to_numpy()
    adata.uns['correction_cell_list'] = list(overlap_matrix.index)
    return

def get_molecule_normalization_factors(
    intensities_df: pd.DataFrame,
    overlap_matrix: pd.DataFrame,
    method: Callable[..., float]
    ) -> Tuple[pd.Series, pd.Series]:
    """Calculates two series with pixel- and molecule-specific normalization factors
    pixels_total_overlap: pixel-specific sampling proportion (the fraction of a pixel
    that overlaps with cells)
    full_pixels_avg_intensities: molecule-specific average intensity of all pixels
    with complete cell overlap

    Args:
        intensities_df (pd.DataFrame): [description]
        overlap_matrix (pd.DataFrame): [description]
        method (Callable[..., float]): [description]

    Returns:
        Tuple[pd.Series, pd.Series]: pixels_total_overlap and full_pixels_avg_intensities
    """
    # sum up all cellular overlaps of each pixel
    pixels_total_overlap = overlap_matrix.sum(axis=0)
    
    # get all pixels whose cellular overlaps sum up to the total pixel area (1)
    full_pixels = pd.Series(pixels_total_overlap[pixels_total_overlap == 1]).index
        # iterate over all metabolites and take average (or other measure) of intensities in full pixels
    full_pixels_avg_intensities = intensities_df.apply(lambda x: method(x[full_pixels]))

    # return both series
    return (pixels_total_overlap, full_pixels_avg_intensities)

def add_normalization_factors(adata: ad.AnnData,
    method: Callable[..., float]) -> None:
    """Calculates two sets of normalization factors and attaches them to the given AnnData object.
    The object is being manipulated inplace.

    Args:
        adata (ad.AnnData): Pixel-based AnnData object
        method (Callable[..., float]): method to use for calculation of full_pixel_avg_intensities
    """

    overlap, full_pix = get_molecule_normalization_factors(intensities_df=adata.to_df(),
        overlap_matrix = adata.obsm['correction_overlap_matrix'].T,
        method = method)
    adata.obs['correction_total_pixel_overlap'] = overlap
    adata.var['correction_full_pixel_avg_intensities'] = full_pix
    return


def normalize_proportion_ratios(
    intensities_df: pd.DataFrame,
    pixels_total_overlap: pd.Series,
    full_pixels_avg_intensities: pd.Series,
    normalized = True
    ) -> pd.DataFrame:
    """Calculates intensity / sampling proportion rates for ion intensities

    Args:
        intensities_df (pd.DataFrame): [description]
        pixels_total_overlap (pd.Series): [description]
        full_pixels_avg_intensities (pd.Series): [description]
        normalized (bool, optional): [description]. Defaults to True.

    Returns:
        pd.DataFrame: [description]
    """
    if(len(intensities_df) != len(pixels_total_overlap) or \
       len(intensities_df.columns) != len(full_pixels_avg_intensities)):
        print('normalize_proportion_ratios: Inconsistant size of arguments. Coercing')

    # calculate intensity / sampling proportion ratios
    intensity_prop_ratios = intensities_df.divide(
        pixels_total_overlap.replace(to_replace=0, value=np.nan), axis = 0)

    # normalize ratios so that full pixels have intensities with avg = 1
    norm_intensity_prop_ratios = intensity_prop_ratios.divide(
        full_pixels_avg_intensities, axis=1).replace(np.inf, np.nan)

    # normalization is optional
    if normalized:
        return norm_intensity_prop_ratios
    return intensity_prop_ratios


def correct_intensities_quantile_regression(
    intensities_df: pd.DataFrame,
    pixels_total_overlap: pd.Series,
    full_pixels_avg_intensities: pd.Series,
    reference_ions: list,
    proportion_threshold = 0.1
    ) -> pd.DataFrame:
    """Corrects ion intensities based on cell sampling proportion of respective pixels

    Args:
        intensities_df (pd.DataFrame): Ion intensity DataFrame with molecules in columns and
        pixels in rows
        pixels_total_overlap (pd.Series): [description]
        full_pixels_avg_intensities (pd.Series): [description]
        proportion_threshold (float, optional): Defaults to 0.1.

    Returns:
        pd.DataFrame: [description]
    """
    min_datapoints = 10

    # get Series name of pixel sampling proportions for model formula
    reference = pixels_total_overlap.name

    if len(intensities_df) != len(pixels_total_overlap):
        print('Quantreg: Inconsistent sizes of arguments')

    # calculate intensity / sampling proportion ratios
    prop_ratio_df = normalize_proportion_ratios(intensities_df=intensities_df,
        pixels_total_overlap=pixels_total_overlap,
        full_pixels_avg_intensities=full_pixels_avg_intensities)

    # take log of both variables: intensity / sampling proportion ratios and sampling proportions
    log_prop_series = np.log10(pixels_total_overlap)
    log_ratio_df = np.log10(prop_ratio_df.replace(np.nan, 0).infer_objects())
    log_ratio_df = log_ratio_df.replace([np.inf, - np.inf], np.nan)

    reference_df = pd.concat([log_ratio_df[reference_ions], log_prop_series], axis=1) \
        .melt(id_vars=(reference))[['value', reference]]
    
    reference_df = reference_df[reference_df[reference] > np.log10(proportion_threshold)].dropna()

    if len(reference_df) < min_datapoints:
        raise RuntimeError("The supplied reference pool has only %1d valid data points and is "
        + "therefore unsuitable. Please specify a suitable reference pool."%len(reference_df))

    ref_model = smf.quantreg('Q("value") ~ ' + reference, reference_df)
    ref_qrmodel = ref_model.fit(q=0.5)

    # create output variables
    correction_factors = log_ratio_df.copy().applymap(lambda x: np.nan)
    params = {}
    predictions = log_ratio_df.copy().applymap(lambda x: np.nan)

    insufficient_metabolites_list = []

    # iterate over molecules
    for ion in log_ratio_df.columns:
        # for every molecule, create custom df with two regression variables
        ion_df = pd.concat([log_ratio_df[ion], log_prop_series], axis=1)
        # filter data for model fitting: only include pixels with given sampling proportion
        # threshold and remove NAs (caused by zero intensities)
        df_for_model = ion_df[ion_df[reference] > np.log10(proportion_threshold)].dropna()

        # check if enough data remains after filtering, otherweise what TODO?
        if len(df_for_model) < 10:
            #print('%s has only %1d, thus not enough datapoints. using reference pool instead'%(ion, len(df_for_model)))
            qrmodel=ref_qrmodel
            insufficient_metabolites_list.append(ion)

        else: 
            # calculate quantile regression
            model = smf.quantreg('Q("' + ion + '") ~ ' + reference, df_for_model)
            qrmodel = model.fit(q=0.5)
            params[ion] = qrmodel.params

        # regress out dependency on sampling proportion
        reg_correction = 10 ** qrmodel.predict(ion_df)
        predictions[ion] = intensities_df[ion] / reg_correction
        correction_factors[ion] = reg_correction
        
    # print(pd.concat([ion_intensities['C16H30O2'], sampling_proportion_series, log_ratio_df['C16H30O2'], correction_factors['C16H30O2'], predictions['C16H30O2']], axis=1))
    print('insufficient metabolites: %1d'%len(insufficient_metabolites_list))
    #print(insufficient_metabolites_list)
    #return((correction_factors, pd.Series(params), predictions))
    return predictions


def correct_intensities_quantile_regression_parallel(
    intensities_ad: ad.AnnData,
    pixels_total_overlap: pd.Series,
    full_pixels_avg_intensities: pd.Series,
    reference_ions: list,
    proportion_threshold = 0.1,
    min_datapoints = 10,
    n_jobs = 1
    ) -> ad.AnnData:
    """Corrects ion intensities based on cell sampling proportion of respective pixels

    Args:
        intensities_ad (ad.AnnData): Ion intensity DataFrame with molecules in columns and
        pixels in rows
        pixels_total_overlap (pd.Series): [description]
        full_pixels_avg_intensities (pd.Series): [description]
        proportion_threshold (float, optional): [description]. Defaults to 0.1.

    Returns:
        ad.AnnData: [description]
    """
    
    # get Series name of pixel sampling proportions for model formula
    pixels_total_overlap = pd.Series(pixels_total_overlap, name='pixels_total_overlap')
    reference = pixels_total_overlap.name

    if len(intensities_ad) != len(pixels_total_overlap):
        print('Quantreg: Inconsistent sizes of arguments')

    # calculate intensity / sampling proportion ratios
    prop_ratio_df = normalize_proportion_ratios(intensities_df=intensities_ad.to_df(),
        pixels_total_overlap=pixels_total_overlap,
        full_pixels_avg_intensities=full_pixels_avg_intensities)

    # take log of both variables: intensity / sampling proportion ratios and sampling proportions
    log_prop_series = np.log10(pixels_total_overlap.replace(0, np.nan))
    log_ratio_df = np.log10(prop_ratio_df.replace(0, np.nan))
    
    # log_prop_series = np.log10(pixels_total_overlap)
    # log_ratio_df = np.log10(prop_ratio_df.replace(np.nan, 0).infer_objects())
    # log_ratio_df = log_ratio_df.replace([np.inf, - np.inf], np.nan)

    reference_df = pd.concat([log_ratio_df[reference_ions], log_prop_series], axis=1) \
        .melt(id_vars=(reference))[['value', reference]]
    
    reference_df = reference_df[reference_df[reference] > np.log10(proportion_threshold)].dropna()

    if len(reference_df) < min_datapoints:
        raise RuntimeError("The supplied reference pool has only %1d valid data points and is "
        + "therefore unsuitable. Please specify a suitable reference pool."%len(reference_df))

    ref_model = smf.quantreg('Q("value") ~ ' + reference, reference_df)
    ref_qrmodel = ref_model.fit(q=0.5)

    def quantile_ion(ion):
        # for every molecule, create custom df with two regression variables
        ion_df = pd.concat([log_ratio_df[ion], log_prop_series], axis=1)
        # filter data for model fitting: only include pixels with given sampling proportion
        # threshold and remove NAs (caused by zero intensities)
        df_for_model = ion_df[ion_df[reference] > np.log10(proportion_threshold)].dropna()

        # check if enough data remains after filtering
        if len(df_for_model) < min_datapoints:
            #print('%s has only %1d, thus not enough datapoints. using reference pool instead'%(ion, len(df_for_model)))
            qrmodel=ref_qrmodel

        else:
            # calculate quantile regression
            model = smf.quantreg('Q("' + ion + '") ~ ' + reference, df_for_model)
            qrmodel = model.fit(q=0.5)

        # regress out dependency on sampling proportion
        reg_correction = 10 ** qrmodel.predict(ion_df)
        
        raw = intensities_ad.to_df()[ion]
        raw.name = "raw"
        ion_df = ion_df.join(raw)
        ion_df['raw_corrected'] = ion_df['raw'] / reg_correction
        ion_df['corrected'] = ion_df[ion] / reg_correction
        ion_df[reference] = pixels_total_overlap
        
        pred = ion_df['raw_corrected']
        pred.name = ion
        return (pred, len(df_for_model), qrmodel.iterations, qrmodel.params)

    # iterate over molecules
    predictions_tuples = Parallel(n_jobs=n_jobs)(
        delayed(quantile_ion)(ion) for ion in log_ratio_df.columns)
    predictions_dict = {i[0].name: i[0] for i in predictions_tuples}
    datapoints_list = [i[1] for i in predictions_tuples]
    iterations_list = [i[2] for i in predictions_tuples]
    slope_list = [i[3][1] for i in predictions_tuples]
    predictions_ad = intensities_ad.copy()

    predictions_ad.X = pd.DataFrame(predictions_dict).replace(np.nan, 0)
    predictions_ad.var['correction_n_datapoints'] = datapoints_list
    predictions_ad.var['correction_using_ion_pool'] = [datapoint < min_datapoints for datapoint in datapoints_list]
    predictions_ad.var['correction_n_iterations'] = iterations_list
    predictions_ad.var['correction_quantreg_slope'] = slope_list
    predictions_ad.obs['total_pixel_overlap'] = pixels_total_overlap
    
    # print(pd.concat([ion_intensities['C16H30O2'], sampling_proportion_series, log_ratio_df['C16H30O2'], correction_factors['C16H30O2'], predictions['C16H30O2']], axis=1))
    #print(insufficient_metabolites_list)
    #return((correction_factors, pd.Series(params), predictions))
    return predictions_ad

def correct_quantile_inplace(adata: ad.AnnData,
    reference_ions: list,
    proportion_threshold = 0.1,
    min_datapoints = 10,
    n_jobs = 1
) -> ad.AnnData:

    an = correct_intensities_quantile_regression_parallel(intensities_ad = adata, 
        pixels_total_overlap = adata.obs['correction_total_pixel_overlap'],
        full_pixels_avg_intensities = adata.var['correction_full_pixel_avg_intensities'],
        reference_ions = reference_ions,
        proportion_threshold = proportion_threshold,
        min_datapoints = min_datapoints,
        n_jobs = n_jobs
        )

    return an


def cell_normalization_Rappez_adata(sampling_prop_matrix: pd.DataFrame,
    sampling_spec_matrix: pd.DataFrame,
    adata: ad.AnnData,
    raw_adata: ad.AnnData,
    sampling_prop_threshold = 0.3,
    sampling_spec_threshold = 0
) -> ad.AnnData:
    
    # filter out pixels with little overlap with any cell (thus sum of all overlaps)
    pixel_sampling_prop_keep = sampling_prop_matrix.sum(axis = 0) > sampling_prop_threshold
    # filter out pixels with low contributions to a cell
    pixel_sampling_spec_keep = sampling_spec_matrix > sampling_spec_threshold

    sampling_prop_matrix_filtered = sampling_prop_matrix.sum(axis = 0) * pixel_sampling_prop_keep
    sampling_spec_matrix_filtered = sampling_spec_matrix * pixel_sampling_spec_keep
    # sampling_spec_matrix_filtered[sampling_spec_matrix_filtered == 0] = np.nan

    sum_prop_matrix = sampling_prop_matrix_filtered.astype(float)#.replace(to_replace=0, value=np.nan)
    sum_prop_matrix[sum_prop_matrix == 0] = np.nan

    # create dataframe for results
    norm_ion_intensities = ad.AnnData(obs=pd.DataFrame({'cell_id': adata.uns['correction_cell_list']},
                                                        index=adata.uns['correction_cell_list']),
                                      var=adata.var)

    norm_spots = adata.to_df().multiply(1/sum_prop_matrix, axis=0)
    # making sure the matrices for matrix product dont contain nans
    sampling_spec_matrix_filtered[np.isnan(sampling_spec_matrix_filtered)] = 0
    norm_spots[np.isnan(norm_spots)] = 0

    cor_df = sampling_spec_matrix_filtered.dot(norm_spots)
    inv_sampling_spec = 1/sampling_spec_matrix_filtered.sum(axis=1)
    norm_ion_intensities.X = (cor_df * inv_sampling_spec[:, np.newaxis]).astype(np.float32)
    norm_ion_intensities.obs.index = norm_ion_intensities.obs.cell_id.map(lambda x: x.replace(CELL_PRE, ""))

    norm_ion_intensities = norm_ion_intensities[:, raw_adata.var_names]
    norm_ion_intensities = norm_ion_intensities[raw_adata.obs_names]
    norm_ion_intensities.obs = raw_adata.obs
    
    return norm_ion_intensities


def deconvolution_rappez(adata: ad.AnnData,
    raw_adata: ad.AnnData = None,
    sampling_prop_threshold = 0.3,
    sampling_spec_threshold = 0
) -> ad.AnnData:

    if raw_adata is None:
        raw_adata = ad.AnnData(var=adata.var, 
            obs=pd.DataFrame({'cell_id':adata.obsm['correction_overlap_matrix'].columns}, 
                index=adata.obsm['correction_overlap_matrix'].columns))

    deconv_adata = cell_normalization_Rappez_adata(
        sampling_prop_matrix = adata.obsm['correction_overlap_matrix'].T,
        sampling_spec_matrix = adata.obsm['correction_sampling_spec_matrix'].T,
        adata = adata,
        raw_adata = raw_adata,
        sampling_prop_threshold = sampling_prop_threshold,
        sampling_spec_threshold = sampling_spec_threshold
    )

    return deconv_adata


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


def intermixing(adata1, adata2, condition_name, labels: Tuple[str, str], show_table=[10], sample_frac=0.2):
    fig, ax = plt.subplots(1, 2)
    summaries = {}

    summaries['umap_1'] = intermixing_metric_sampled(adata1, condition_name, sample_frac=sample_frac, 
        ax=ax[0], label=labels[0]+'_umap', measure='X_umap', normalized=True)
    summaries['umap_2'] = intermixing_metric_sampled(adata2, condition_name, sample_frac=sample_frac, 
        ax=ax[0], label=labels[1]+'_umap', measure='X_umap', normalized=True)

    summaries['pca_1'] = intermixing_metric_sampled(adata1, condition_name, sample_frac=sample_frac, 
        ax=ax[1], label=labels[0]+'_pca', measure='X_pca', normalized=True)
    summaries['pca_2'] = intermixing_metric_sampled(adata2, condition_name, sample_frac=sample_frac, 
        ax=ax[1], label=labels[1]+'_pca', measure='X_pca', normalized=True)

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    
    if all([hood in summaries['umap_1'].index for hood in show_table]):
        return pd.concat({k: v.loc[show_table] for k, v in summaries.items()})
    else:
        return pd.concat([intermixing_metric_sampled(adata1, condition_name=condition_name, 
            neighborhood_size=show_table, n_datapoints=1, label='none', normalized=True),
            intermixing_metric_sampled(adata2, condition_name=condition_name, 
            neighborhood_size=show_table, n_datapoints=1, label='ISM', normalized=True)], 
            axis=0, keys=[labels[0], labels[1]])