"""Functions for the calculation of corrected ion intensity data
to mitigate effects of ion suppression

Functions:

Author: Marius Klein (mklein@duck.com), October 2022
"""
from typing import Tuple, Callable
import warnings
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy.builtins import Q 
from tqdm import tqdm
import anndata as ad
import scanpy as sc
# import sys
# sys.path.append('/home/mklein/spacem')
from SpaceM.lib.modules import (
    overlap_analysis,
    single_cell_analysis_normalization
)
from ISC.src import const


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
    overlap_matrix = pd.DataFrame(index=[const.CELL_PRE + str(n) for n in cell_area.cell_id.astype(int)],
                                             columns=[const.PIXEL_PRE + str(n) for n in mark_area.am_id.astype(int)])

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
            overlap_matrix.loc[const.CELL_PRE + str(int(cell_i)), const.PIXEL_PRE + str(int(pixel_row['am_id']))] = pixel_row['area']

    total_pixel_size = pd.Series(mark_area.area).replace(0, np.nan)
    total_pixel_size.index = overlap_matrix.columns

    sampling_prop_matrix = overlap_matrix.divide(total_pixel_size, axis=1).replace(np.nan, 0)

    total_cell_coverage = overlap_matrix.sum(axis=1).replace(to_replace=0, value=1)
    sampling_spec_matrix = overlap_matrix.divide(total_cell_coverage, axis=0).replace(np.nan, 0)

    return(sampling_prop_matrix, sampling_spec_matrix)


def add_matrices(
    adata: ad.AnnData,
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
    adata: pd.DataFrame,
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
    
    overlap_matrix = adata.obsm['correction_overlap_matrix'].T
    intensities_df = adata.to_df()
    
    # sum up all cellular overlaps of each pixel
    pixels_total_overlap = overlap_matrix.sum(axis=0) / np.array(adata.obs['area'])
    
    # iterate over all metabolites and take average (or other measure) of intensities in full pixels
    intensities_df[const.TPO] = pixels_total_overlap

    full_intensities_df = intensities_df[intensities_df[const.TPO] == 1]
    
    if len(full_intensities_df) == 0:
        # threshold = np.percentile(intensities_df[const.TPO], 99)
        arr = intensities_df[const.TPO].values
        threshold = float(arr[np.argsort(arr)][-10:-9])

        full_intensities_df = intensities_df[intensities_df[const.TPO] >= threshold]
        print('No pixels with full overlap found. Taking top ten pixels with overlap > %1.2f'%threshold)
        
    del full_intensities_df[const.TPO]
    print('Using %d pixels to calculate full-pixel avereage intensities.'%len(full_intensities_df))
    full_pixels_avg_intensities = np.array([method(full_intensities_df[ion]) for ion in full_intensities_df])

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

    overlap, full_pix = get_molecule_normalization_factors(adata=adata,
        method = method)
    adata.obs[const.TPO] = overlap
    adata.var[const.FPAI] = full_pix
    return

def normalize_proportion_ratios(
    intensities_ad: ad.AnnData,
    normalized = True
    ) -> ad.AnnData:
    """Calculates intensity / sampling proportion rates for ion intensities

    Args:
        intensities_ad (ad.AnnData): [description]
        normalized (bool, optional): [description]. Defaults to True.

    Returns:
        ad.AnnData: [description]
    """
    
    # first checking if required functions have been called before
    if const.TPO not in intensities_ad.obs.columns:
        raise Exception("get_reference_pool has to be called after add_normalization_factors")
    
    # calculate intensity / sampling proportion ratios
    total_pixel_overlap = np.array(
        intensities_ad.obs[const.TPO]).reshape((len(intensities_ad), 1))
    total_pixel_overlap[total_pixel_overlap == 0] = np.nan

    full_pixels_avg_intensities = np.array(
        intensities_ad.var[const.FPAI]).reshape((1, len(intensities_ad.var)))
    full_pixels_avg_intensities[full_pixels_avg_intensities == 0] = np.nan

    intensity_prop_ratios = intensities_ad.X / total_pixel_overlap

    # normalize ratios so that full pixels have intensities with avg = 1
    norm_intensity_prop_ratios = intensity_prop_ratios / full_pixels_avg_intensities
    
    out_ad = intensities_ad.copy()

    # normalization is optional
    if normalized:
        out_ad.X = norm_intensity_prop_ratios
    else: 
        out_ad.X = intensity_prop_ratios
    
    return out_ad


def get_reference_pool(
    am_adata: ad.AnnData,
    reference_pool_config = None,
    normalized = True
    ) -> list:
    """determines reference ion pool for ions with too few data points

    Args:
        am_adata (ad.AnnData): annadata on ablation mark level
        reference_pool_config (optional): Dictionary with measure and number keys to specify the 
        way reference ions are selected. Defaults to None. If an unrecognized measure is given,
        all ions are returned as reference pool.
        normalized (optional bool): determines whether intensity proportion ratios should be 
        normalized to 1 at full overlap

    Returns:
        list of reference ions. 
    """
    reference_pool = am_adata.var_names
    
    if reference_pool_config is None:
        reference_pool_config = {'measure': 'max_datapoints',
                                 'number': 10}

    ratio_df = normalize_proportion_ratios(am_adata, normalized=normalized).to_df().replace(np.nan, 0)
    
    if reference_pool_config['measure'] == "max_datapoints":
        reference_pool = ratio_df.astype(bool).sum(axis=0).sort_values().tail(reference_pool_config['number']).index
        
    return list(reference_pool)



def correct_intensities_quantile_regression_parallel(
    intensities_ad: ad.AnnData,
    pixels_total_overlap: pd.Series,
    full_pixels_avg_intensities: pd.Series,
    reference_ions: list,
    proportion_threshold = 0.1,
    min_datapoints = 10,
    correct_intersect = False,
    normalized = True,
    n_jobs = 1,
    progress = False
    ) -> ad.AnnData:
    """Corrects ion intensities based on cell sampling proportion of respective pixels

    Args:
        intensities_ad (ad.AnnData): Ion intensity DataFrame with molecules in columns and
        pixels in rows
        pixels_total_overlap (pd.Series): [description]
        full_pixels_avg_intensities (pd.Series): [description]
        proportion_threshold (float, optional): [description]. Defaults to 0.1.
        min_datapoints (int, optional): number of datapoints required for correction using
        this molecule. Otherwise falling back to reference ions.
        correct_intersect (bool, optional): whether to correct for an intersect not equal to
        zero. This constant scales the whole set of intensities independant from the 
        sampling ratio.
        normalized (optional bool): determines whether intensity proportion ratios should be 
        normalized to 1 at full overlap
        n_jobs (int, optional): number of cores to use for parallel processing.

    Returns:
        ad.AnnData: Annotated data matrix with corrected intensities assigned.
    """
    
    # get Series name of pixel sampling proportions for model formula
    pixels_total_overlap = pd.Series(pixels_total_overlap, name='pixels_total_overlap')
    reference = pixels_total_overlap.name

    if len(intensities_ad) != len(pixels_total_overlap):
        print('Quantreg: Inconsistent sizes of arguments')

    # calculate intensity / sampling proportion ratios
    prop_ratio_ad = normalize_proportion_ratios(intensities_ad=intensities_ad, normalized=normalized)

    # take log of both variables: intensity / sampling proportion ratios and sampling proportions
    log_prop_series = np.log10(pixels_total_overlap.replace(0, np.nan))
    log_ratio_df = np.log10(prop_ratio_ad.to_df().replace(0, np.nan))
    
    # log_prop_series = np.log10(pixels_total_overlap)
    # log_ratio_df = np.log10(prop_ratio_df.replace(np.nan, 0).infer_objects())
    # log_ratio_df = log_ratio_df.replace([np.inf, - np.inf], np.nan)

    reference_df = pd.concat([log_ratio_df[reference_ions], log_prop_series], axis=1) \
        .melt(id_vars=(reference))[['value', reference]]
    
    reference_df = reference_df[
        reference_df[reference] > np.log10(proportion_threshold)
    ].dropna()

    if len(reference_df) < min_datapoints:
        raise RuntimeError(f"The supplied reference pool has only {len(reference_df)} valid data " +
        "points and is therefore unsuitable. Please specify a suitable reference pool.")

    warnings.filterwarnings("ignore")
    
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
        ion_df['raw_corrected_intersect'] = ion_df['raw'] / reg_correction * 10 ** qrmodel.params[0]
        ion_df['corrected'] = ion_df[ion] / reg_correction
        ion_df[reference] = pixels_total_overlap
        
        pred = ion_df['raw_corrected']
        if correct_intersect:
            pred = ion_df['raw_corrected_intersect']
        pred.name = ion
        return (pred, len(df_for_model), qrmodel.iterations, qrmodel.params)

    # iterate over molecules
    if progress:
        predictions_tuples = Parallel(n_jobs=n_jobs)(
            delayed(quantile_ion)(ion) for ion in tqdm(log_ratio_df.columns))
    else:
         predictions_tuples = Parallel(n_jobs=n_jobs)(
            delayed(quantile_ion)(ion) for ion in log_ratio_df.columns)
    
    # extracting all the important types of data
    predictions_dict = {i[0].name: i[0] for i in predictions_tuples}
    datapoints_list = [i[1] for i in predictions_tuples]
    iterations_list = [i[2] for i in predictions_tuples]
    slope_list = [i[3][1] for i in predictions_tuples]
    intersect_list = [i[3][0] for i in predictions_tuples]
    predictions_ad = intensities_ad.copy()

    # saving gathed data into AnnData object
    predictions_ad.X = pd.DataFrame(predictions_dict).replace(np.nan, 0)
    predictions_ad.var['correction_n_datapoints'] = datapoints_list
    predictions_ad.var['correction_using_ion_pool'] = [datapoint < min_datapoints for datapoint in datapoints_list]
    predictions_ad.var['correction_n_iterations'] = iterations_list
    predictions_ad.var['correction_quantreg_slope'] = slope_list
    predictions_ad.var['correction_quantreg_intersect'] = intersect_list
    predictions_ad.obs['total_pixel_overlap'] = pixels_total_overlap
    
    warnings.filterwarnings("default")

    return predictions_ad

def correct_quantile_inplace(adata: ad.AnnData,
    reference_ions: list,
    proportion_threshold = 0.1,
    min_datapoints = 10,
    correct_intersect = False,
    normalized = True,
    n_jobs = 1,
    progress = False,
) -> ad.AnnData:
    """Corrects ion intensities based on cell sampling proportion of respective pixels and 
    returns resulting annotated data matrix.

    Args:
        adata (ad.AnnData): Ion intensity DataFrame with molecules in columns and
        pixels in rows
        reference_ions (list): list with the names of ions used as fallback if correction
        cannot be calculated for molecules using existing data. If more than one molecules 
        are given, all their datapoints are used.
        proportion_threshold (float, optional): Threshold of sampling proportion. Only 
        ablation marks with a higher proportion are used to calculate the correction.
        Defaults to 0.1.
        min_datapoints (int, optional): number of datapoints required for correction using
        this molecule. Otherwise falling back to reference ions.
        correct_intersect (bool, optional): whether to correct for an intersect not equal to
        zero. This constant scales the whole set of intensities independant from the 
        sampling ratio.
        normalized (optional bool): determines whether intensity proportion ratios should be 
        normalized to 1 at full overlap
        n_jobs (int, optional): number of cores to use for parallel processing.

    Returns:
        ad.AnnData: Annotated data matrix with corrected intensities assigned.
    """
    

    an = correct_intensities_quantile_regression_parallel(intensities_ad = adata, 
        pixels_total_overlap = adata.obs[const.TPO],
        full_pixels_avg_intensities = adata.var[const.FPAI],
        reference_ions = reference_ions,
        proportion_threshold = proportion_threshold,
        min_datapoints = min_datapoints,
        correct_intersect = correct_intersect,
        normalized = normalized,
        n_jobs = n_jobs,
        progress = progress
        )

    return an


def get_overlap_data(
    cell_regions: pd.DataFrame,
    mark_regions: pd.DataFrame,
    overlap_regions: pd.DataFrame
) -> overlap_analysis.OverlapData:
    """wrapper around SpaceM dataclass OverlapData

    Args:
        cell_regions (pd.DataFrame): Dataframe with spatial data on cells
        mark_regions (pd.DataFrame): Dataframe with spatial data on ablation marks
        overlap_regions (pd.DataFrame): Dataframe with spatial data on overlap regions

    Returns:
        OverlapData: dataclass that contains the relevant overlap data for SpaceM deconvolution
    """
    return overlap_analysis.OverlapData(
        cell_regions=cell_regions.set_index('cell_id'),
        ablation_mark_regions=mark_regions.set_index('am_id'),
        overlap_regions=overlap_regions,
        overlap_labels=None
    )


def add_overlap_matrix_spacem(
    adata: ad.AnnData,
    cell_regions: pd.DataFrame,
    mark_regions: pd.DataFrame,
    overlap_regions: pd.DataFrame
):
    """wrapper around SpaceMs calculate_overlap_matrix and adding this data to AnnData object 

    Args:
        adata (ad.AnnData): annotated data matrix to add overlap data to.
        cell_regions (pd.DataFrame): Dataframe with spatial data on cells
        mark_regions (pd.DataFrame): Dataframe with spatial data on ablation marks
        overlap_regions (pd.DataFrame): Dataframe with spatial data on overlap regions
    """

    overlap_data = get_overlap_data(cell_regions, mark_regions, overlap_regions)

    overlap_matrix = overlap_analysis.compute_overlap_matrix(
        overlap_data.overlap_regions,
        overlap_data.cell_regions.index,
        overlap_data.ablation_mark_regions.index,
    )

    # prepare data that is required for the quantile regression
    adata.obsm['correction_overlap_matrix'] = np.array(overlap_matrix.T)
    adata.obs['area'] = list(mark_regions.area)
    adata.uns['correction_cell_list'] = list(overlap_matrix.index)
    adata.uns['cell_area'] = list(cell_regions.area)


def deconvolution_spacem(
    adata: ad.AnnData,
    overlap_data: overlap_analysis.OverlapData,
    raw_adata: ad.AnnData = None,
    deconvolution_params: dict = None
) -> ad.AnnData:
    """AI is creating summary for deconvolution_spacem

    Args:
        adata (ad.AnnData): ablation-mark based adata to be deconvoluted
        overlap_data (overlap_analysis.OverlapData): corresponding overlap data
        raw_adata (ad.AnnData, optional): cell-based adata that is used as prototype to deconvolute
        adata. Defaults to None.
        deconvolution_params (dict, optional): Dictionary with keys 'cell_normalization_method' and
        'ablation_marks_min_overlap_ratio' to guide SpaceMs deconvolution. Default values are
         * cell_normalization_method:         'weighted_by_overlap_and_sampling_area'
         * ablation_marks_min_overlap_ratio:  0.3

    Returns:
        ad.AnnData: deconvoluted adata
    """
    if raw_adata is None:
        raw_adata = ad.AnnData(var = adata.var, obs = pd.DataFrame(index=adata.uns['correction_cell_list']))
        
    if deconvolution_params is None:
        deconvolution_params = {'cell_normalization_method': 'weighted_by_overlap_and_sampling_area', 
                                'ablation_marks_min_overlap_ratio': 0.3
                               }
        print('no deconvolution parameters given!')
    
    metabolites = raw_adata.var_names.intersection(adata.var_names)

    spectra_df = sc.get.obs_df(adata, keys=list(metabolites))
    spectra_df.index = spectra_df.index.astype(int)
    
    cell_spectra = single_cell_analysis_normalization.create_cell_ion_intensities_dataframe_from_spectra_df(
        am_spectra_df=spectra_df,
        overlap_data = overlap_data,
        cell_normalization_method = deconvolution_params['cell_normalization_method'],
        ablation_marks_min_overlap_ratio = deconvolution_params['ablation_marks_min_overlap_ratio']
    )
    cells = cell_spectra.index.astype(str).intersection(raw_adata.obs_names)
    
    cell_adata = raw_adata[cells, metabolites].copy()
    cell_adata.X = cell_spectra.loc[cells.astype(int)].values
    cell_adata.var = cell_adata.var.align(adata.var)[1].loc[metabolites]
    return cell_adata


def cell_normalization_rappez_adata(
    sampling_prop_matrix: pd.DataFrame,
    sampling_spec_matrix: pd.DataFrame,
    adata: ad.AnnData,
    raw_adata: ad.AnnData,
    sampling_prop_threshold = 0.3,
    sampling_spec_threshold = 0,
    remove_empty_cells = False,
) -> ad.AnnData:
    """ deconvolution method implemented after Martijn Molenaars FDA-project code
    """
    
    # filter out pixels with little overlap with any cell (thus sum of all overlaps)
    pixel_sampling_prop_keep = sampling_prop_matrix.sum(axis = 0) > sampling_prop_threshold
    # filter out pixels with low contributions to a cell
    pixel_sampling_spec_keep = sampling_spec_matrix > sampling_spec_threshold

    sampling_prop_matrix_filtered = sampling_prop_matrix.sum(axis = 0) * pixel_sampling_prop_keep
    sampling_spec_matrix_filtered = sampling_spec_matrix * pixel_sampling_spec_keep

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
    cell_norm_factor = sampling_spec_matrix_filtered.sum(axis=1)
    cell_norm_factor[cell_norm_factor == 0] = np.nan
    deconv_array = cor_df / cell_norm_factor[:, np.newaxis]
    deconv_array[np.isnan(deconv_array)] = 0
    norm_ion_intensities.X = deconv_array.astype(np.float32)
    #norm_ion_intensities.obs.index = norm_ion_intensities.obs.cell_id.map(lambda x: x.replace(const.CELL_PRE, ""))

    obs_names = raw_adata.obs_names
    var_names = raw_adata.var_names.intersection(adata.var_names)
    if remove_empty_cells:
        cells_out = np.array(adata.uns['correction_cell_list'])[np.array(adata.obsm['correction_overlap_matrix']).sum(axis=0) != 0]
        obs_names = raw_adata.obs_names.intersection(cells_out)

    norm_ion_intensities = norm_ion_intensities[obs_names, var_names]
    norm_ion_intensities.obs = raw_adata.obs.loc[obs_names, :]
    return norm_ion_intensities

def deconvolution_rappez(
    adata: ad.AnnData,
    raw_adata: ad.AnnData = None,
    sampling_prop_threshold = 0.3,
    sampling_spec_threshold = 0,
    remove_empty_cells = False,
) -> ad.AnnData:
    """Pixel-cell deconvolution as implemented by Martijn Molenaar in FDA-project
    
    Args:
        adata (anndata.AnnData): ablation-mark based annotated data matrix
        raw_adata (ad.AnnData, optional): cell-based adata that is used as prototype to deconvolute
        adata. Defaults to None.
        sampling_prop_threshold (int): Translates to ablation_marks_min_overlap_ratio.
        Defaults to 0.3,
        sampling_spec_threshold (int): Defaults to 0
    
    """

    if raw_adata is None:
        raw_adata = ad.AnnData(var=adata.var, 
            obs=pd.DataFrame({'cell_id':adata.uns['correction_cell_list']}, 
                index=adata.uns['correction_cell_list']))

    deconv_adata = cell_normalization_rappez_adata(
        sampling_prop_matrix = adata.obsm['correction_overlap_matrix'].T,
        sampling_spec_matrix = adata.obsm['correction_sampling_spec_matrix'].T,
        adata = adata,
        raw_adata = raw_adata,
        sampling_prop_threshold = sampling_prop_threshold,
        sampling_spec_threshold = sampling_spec_threshold,
        remove_empty_cells = remove_empty_cells,
    )

    return deconv_adata

