import numpy as np
import pandas as pd
import pytest
from src.correction import CELL_PRE, PIXEL_PRE, get_matrices, get_matrices_from_dfs, get_molecule_normalization_factors, normalize_proportion_ratios


@pytest.fixture
def cell_marks_minimal():
    return {'1':['2', '4'], '2':['3'], '3':[]}

@pytest.fixture
def marks_cell_overlap_minimal():
    return {'1':[10, 5], '2':[5], '3':[]}

@pytest.fixture
def mark_area_minimal():
    return {'1': 10, '2': 10, '3':10, '4':10}

@pytest.fixture
def mark_area_df():
    return pd.DataFrame({'am_id':[1, 2, 3, 4], 'area':[10, 10, 10, 10]})

@pytest.fixture
def cell_area_df():
    return pd.DataFrame({'cell_id':[1, 2, 3]})

@pytest.fixture
def overlap_df():
    return pd.DataFrame({'cell_id':[1, 1, 2], 'am_id':[2, 4, 3], 'area':[10, 5, 5]})

@pytest.fixture
def overlap_matrix():
    df = pd.DataFrame({'1':[np.nan, np.nan, np.nan], '2':[10.0, np.nan, np.nan], '3':[np.nan, 5.0, np.nan], '4':[5.0, np.nan, np.nan] })
    df.columns = [PIXEL_PRE + i for i in df.columns]
    df.index = [CELL_PRE + i for i in ['1', '2', '3']]
    df = df.replace(np.nan, 0)
    return df

@pytest.fixture
def sampling_prop_matrix():
    df = pd.DataFrame({'1':[np.nan, np.nan, np.nan], '2':[1.0, np.nan, np.nan], '3':[np.nan, 0.5, np.nan], '4':[0.5, np.nan, np.nan] })
    df.columns = [PIXEL_PRE + i for i in df.columns]
    df.index = [CELL_PRE + i for i in ['1', '2', '3']]
    df = df.replace(np.nan, 0)
    return df

@pytest.fixture
def sampling_spec_matrix():
    df = pd.DataFrame({'1':[np.nan, np.nan, np.nan], '2':[2/3, np.nan, np.nan], '3':[np.nan, 1.0, np.nan], '4':[1/3, np.nan, np.nan] })
    df.columns = [PIXEL_PRE + i for i in df.columns]
    df.index = [CELL_PRE + i for i in ['1', '2', '3']]
    df = df.replace(np.nan, 0)
    return df
 
@pytest.fixture
def intensities_df():
    df = pd.DataFrame({'ion_1':[0, 10, 10, 5], 'ion_2':[10, 0, 5, 5]})
    df.index = [PIXEL_PRE + i for i in ['1', '2', '3', '4']]
    return df

@pytest.fixture
def pixels_total_overlap():
    ser = pd.Series({'1': 0.0, '2': 1.0, '3': 0.5, '4': 0.5})
    ser.index = [PIXEL_PRE + i for i in ser.index]
    return ser

@pytest.fixture
def full_pixels_avg_intensities():
    return pd.Series({'ion_1': 10.0, 'ion_2': 0.0})


@pytest.fixture
def norm_intensity_prop_ratios():
    return pd.DataFrame({'ion_1': [np.nan, 1.0, 2.0, 1.0], 'ion_2':[np.nan, np.nan, np.nan, np.nan]}, index=[PIXEL_PRE + i for i in ['1', '2', '3', '4']])


def test_get_matrices(mark_area_minimal, cell_marks_minimal, marks_cell_overlap_minimal, sampling_prop_matrix, sampling_spec_matrix):
    
    problems = []

    gen_overlap_matrix, gen_sampling_spec_matrix = get_matrices(mark_area=mark_area_minimal, marks_cell_associations=cell_marks_minimal, marks_cell_overlap=marks_cell_overlap_minimal)
    
    if not gen_overlap_matrix.shape == gen_sampling_spec_matrix.shape == (3, 4):
        problems.append('generated matrices have different sizes')

    if len(sampling_prop_matrix.compare(gen_overlap_matrix)) > 0:
        problems.append('problem with calculation of sampling_prop_matrix \n' + str(sampling_prop_matrix.compare(gen_overlap_matrix)))

    if len(sampling_spec_matrix.compare(gen_sampling_spec_matrix)) > 0:
        problems.append('problem with calculation of sampling_spec_matrix \n' + str(sampling_spec_matrix.compare(gen_sampling_spec_matrix)))

    assert not problems, "errors occured:\n{}".format("\n".join(problems))


def test_get_matrices_from_dfs(mark_area_df, cell_area_df, overlap_df, sampling_prop_matrix, sampling_spec_matrix):
    
    problems = []

    gen_overlap_matrix, gen_sampling_spec_matrix = get_matrices_from_dfs(mark_area=mark_area_df, cell_area=cell_area_df, marks_cell_overlap=overlap_df)
    
    if not gen_overlap_matrix.shape == gen_sampling_spec_matrix.shape == (3, 4):
        problems.append('generated matrices have different sizes')
    
    if len(sampling_prop_matrix.compare(gen_overlap_matrix)) > 0:
        problems.append('problem with calculation of sampling_prop_matrix \n' + str(sampling_prop_matrix.compare(gen_overlap_matrix)))

    if len(sampling_spec_matrix.compare(gen_sampling_spec_matrix)) > 0:
        problems.append('problem with calculation of sampling_spec_matrix \n' + str(sampling_spec_matrix.compare(gen_sampling_spec_matrix)))

    assert not problems, "errors occured:\n{}".format("\n".join(problems))



def test_get_molecule_normalization_factors(intensities_df, sampling_prop_matrix, pixels_total_overlap, full_pixels_avg_intensities):

    problems = []

    gen_pixels_total_overlap, gen_full_pixels_avg_intensities = get_molecule_normalization_factors(intensities_df, sampling_prop_matrix, method=np.mean)
    
    if len(pixels_total_overlap.compare(gen_pixels_total_overlap)) > 0:
        problems.append('problem with calculation of pixels sampling proportion (pixels_total_overlap)\n' + str(pixels_total_overlap.compare(gen_pixels_total_overlap)))
    
    if len(full_pixels_avg_intensities.compare(gen_full_pixels_avg_intensities)) > 0:
        problems.append('problem with calculation of full_pixels_avg_intensities\n' + str(full_pixels_avg_intensities.compare(gen_full_pixels_avg_intensities)))
    
    assert not problems, 'errors occured:\n{}'.format('\n'.join(problems))


def test_normalize_proportion_ratios(intensities_df, pixels_total_overlap, full_pixels_avg_intensities, norm_intensity_prop_ratios):

    gen_prop_ratios_df = normalize_proportion_ratios(intensities_df=intensities_df, pixels_total_overlap=pixels_total_overlap, full_pixels_avg_intensities=full_pixels_avg_intensities)

    assert len(norm_intensity_prop_ratios.compare(gen_prop_ratios_df)) == 0, "error calculating intensity / sampling proportion ratios" + str(norm_intensity_prop_ratios.compare(gen_prop_ratios_df))
