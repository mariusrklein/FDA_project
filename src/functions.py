# IMPORTS
from typing import Dict, Tuple, Callable
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf 


# CONSTANTS

CELL_PRE = 'cell_'
PIXEL_PRE = 'pixel_'


def get_matrices(
    mark_area: Dict[str, list], marks_cell_associations: Dict[str, list], marks_cell_overlap: Dict[str, int]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # prototype pixel x cell matrix for absolute area overlap of all possible pixel-cell combinations
    overlap_matrix = pd.DataFrame(index=[CELL_PRE + n for n in marks_cell_overlap.keys()], columns=[PIXEL_PRE + n for n in mark_area.keys()])

    # analogous matrix for overlap relative to each pixel area (corresponds to ablated region specific sampling proportion)
    sampling_prop_matrix = overlap_matrix.copy()

    # analogous matrix for constitution of every cell
    sampling_spec_matrix = overlap_matrix.copy()


    for cell_i in marks_cell_associations.keys():
        pixels = marks_cell_associations[cell_i]
        # get the index of the pixels as their order is the same in marks_cell_overlap
        for pixel_i, pixel_loc in enumerate(pixels):
    
            overlap_area = marks_cell_overlap[cell_i][pixel_i]
            # write absolute area overlap of current cell-pixel association to respective location in matrix
            overlap_matrix.loc[CELL_PRE + cell_i, PIXEL_PRE + pixel_loc] = overlap_area
    
    total_pixel_size = dict(zip(overlap_matrix.columns, mark_area.values()))
    sampling_prop_matrix = overlap_matrix.divide(total_pixel_size, axis=1)
    
    total_cell_coverage = overlap_matrix.sum(axis=1).replace(to_replace=0, value=1)
    sampling_spec_matrix = overlap_matrix.divide(total_cell_coverage, axis=0)

    return(overlap_matrix, sampling_prop_matrix, sampling_spec_matrix)



def get_molecule_normalization_factors(intensities_df: pd.DataFrame, overlap_matrix: pd.DataFrame, method: Callable[..., float]) -> Tuple[pd.Series, pd.Series]:

    # sum up all cellular overlaps of each pixel
    pixels_total_overlap = overlap_matrix.sum(axis=0).infer_objects()

    # get all pixels whose cellular overlaps sum up to the total pixel area (1)
    full_pixels = pixels_total_overlap[pixels_total_overlap == 1].index

    pixels_total_overlap.name = "total_pixel_area"
    full_pixels.name = "full_pixel_factors_" + method.__name__
    
    # iterate over all metabolites and take average (or other measure) of intensities in full pixels
    full_pixels_avg_intensities = intensities_df.apply(lambda x: method(x[full_pixels]))

    # return both series
    return (pixels_total_overlap, full_pixels_avg_intensities)



def normalize_proportion_ratios(
    intensities_df: pd.DataFrame, pixels_total_overlap: pd.Series, full_pixels_avg_intensities: pd.Series, normalized = True
    ) -> pd.DataFrame:

    if(len(intensities_df) != len(pixels_total_overlap) or len(intensities_df.columns) != len(full_pixels_avg_intensities)):
        print('normalize_proportion_ratios: Inconsistant size of arguments. Coercing using column/index')

    # calculate intensity / sampling proportion ratios
    intensity_prop_ratios = intensities_df.divide(pixels_total_overlap.replace(to_replace=0, value=np.nan), axis = 0)

    # normalize ratios so that full pixels have intensities with avg = 1
    norm_intensity_prop_ratios = intensity_prop_ratios.divide(full_pixels_avg_intensities, axis=1).replace(np.inf, np.nan)
    
    # normalization is optional
    if(normalized): return norm_intensity_prop_ratios
    else: return intensity_prop_ratios



def correct_intensities_quantile_regression(
    intensities_df: pd.DataFrame, pixels_total_overlap: pd.Series, full_pixels_avg_intensities: pd.Series, proportion_threshold = 0.1
    ) -> pd.DataFrame:
    
    # get Series name of pixel sampling proportions for model formula
    reference = pixels_total_overlap.name

    if len(intensities_df) != len(pixels_total_overlap):
        print('Quantreg: Inconsistent sizes of arguments')

    # calculate intensity / sampling proportion ratios
    prop_ratio_df = normalize_proportion_ratios(intensities_df=intensities_df, pixels_total_overlap=pixels_total_overlap, full_pixels_avg_intensities=full_pixels_avg_intensities)
    
    # take logarithm of both variables: intensity / sampling proportion ratios and sampling proportions
    log_prop_series = np.log10(pixels_total_overlap)
    log_ratio_df = np.log10(prop_ratio_df.replace(np.nan, 0).infer_objects())
    log_ratio_df = log_ratio_df.replace([np.inf, - np.inf], np.nan)

    # create output variables
    correction_factors = log_ratio_df.copy().applymap(lambda x: np.nan)
    params = {}
    predictions = log_ratio_df.copy().applymap(lambda x: np.nan)

    # iterate over molecules
    for ion in log_ratio_df.columns:
        # for every molecule, create custom df with two regression variables
        df = pd.concat([log_ratio_df[ion], log_prop_series], axis=1)
        # filter data for model fitting: only include pixels with given sampling proportion threshold and remove NAs (caused by zero intensities)
        df_for_model = df[df[reference] > np.log10(proportion_threshold)].dropna()

        # check if enough data remains after filtering, otherweise what TODO?
        if len(df_for_model) < 3: 
            print('skipping ' + ion)
            continue

        # calculate quantile regression
        model = smf.quantreg(ion + ' ~ ' + reference, df_for_model)
        qrmodel = model.fit(q=0.5)
        params[ion] = qrmodel.params

        # regress out dependency on sampling proportion
        reg_correction = 10 ** qrmodel.predict(df)
        predictions[ion] = intensities_df[ion] / reg_correction 
        correction_factors[ion] = reg_correction
    
    # print(pd.concat([ion_intensities['C16H30O2'], sampling_proportion_series, log_ratio_df['C16H30O2'], correction_factors['C16H30O2'], predictions['C16H30O2']], axis=1))

    #return((correction_factors, pd.Series(params), predictions))
    return predictions
