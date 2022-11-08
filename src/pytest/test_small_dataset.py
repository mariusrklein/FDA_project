import os
import platform
import pytest
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import statistics as st
import multiprocessing
from importlib import reload
import sys
sys.path.append('/Volumes/mklein/FDA_project')
import src.correction


if platform.system() == "Darwin":
    source_path = '/Volumes/alexandr/smenon/2022-07-13_Glioblastoma/processed_files'
    target_path = '/Volumes/mklein/FDA_project/src/pytest'
else:
    source_path = '/g/alexandr/smenon/2022-07-13_Glioblastoma/processed_files'
    target_path = '/home/mklein/FDA_project/src/pytest'

sample = 'B1'
files = {
    'config': '../config.json',
    'sm_matrix': 'ablation_mark_analysis/spatiomolecular_adata.h5ad',
    'overlap_regions': 'overlap_analysis1/overlap.regions.csv',
    'mark_regions': 'overlap_analysis1/ablation_mark.regions.csv',
    'cell_regions': 'overlap_analysis1/cell.regions.csv',
    'cell_sm_matrix': 'single_cell_analysis/spatiomolecular_adata.h5ad',
}

sample_path = os.path.join(source_path, sample, "analysis")

project_files = {k: os.path.join(sample_path, v) for k, v in files.items()}

def get_sm_matrix():
    return sc.read(os.path.join(target_path, 'am_spatiomolecular_adata.h5ad'))

def get_corr_sm_matrix():
    return sc.read(os.path.join(target_path, 'am_spatiomolecular_adata_corrected.h5ad'))

def get_cell_sm_matrix():
    return sc.read(os.path.join(target_path, 'cell_spatiomolecular_adata.h5ad'))

def get_corr_cell_sm_matrix():
    return sc.read(os.path.join(target_path, 'cell_spatiomolecular_adata_corrected.h5ad'))

@pytest.fixture(scope="session")
def sm_matrix():
    return get_sm_matrix()

@pytest.fixture(scope="session")
def corr_sm_matrix():
    return get_corr_sm_matrix()

@pytest.fixture(scope="session")
def cell_sm_matrix():
    return get_cell_sm_matrix()

@pytest.fixture(scope="session")
def corr_cell_sm_matrix():
    return get_corr_cell_sm_matrix()


def get_matrices():
    from src.correction import get_matrices_from_dfs

    cell_regions = pd.read_csv(project_files['cell_regions'])
    mark_regions = pd.read_csv(project_files['mark_regions'])
    overlap_regions = pd.read_csv(project_files['overlap_regions'])
    
    overlap_regions = overlap_regions[:20]
    cell_regions = cell_regions[cell_regions.cell_id.isin(overlap_regions['cell_id'])]
    mark_regions = mark_regions[mark_regions.am_id.isin(overlap_regions['am_id'])]

    overlap_matrix, sampling_spec_matrix = get_matrices_from_dfs(mark_area = mark_regions, cell_area = cell_regions, marks_cell_overlap = overlap_regions)
    return (overlap_matrix, sampling_spec_matrix)

@pytest.fixture
def matrizes():
    return get_matrices()


def create_small_dataset():
    from src.correction import PIXEL_PRE

    sm_matrix = sc.read(os.path.join(sample_path, files['sm_matrix']))
    cell_sm_matrix = sc.read(os.path.join(sample_path, files['cell_sm_matrix']))

    overlap_matrix, sampling_spec_matrix = get_matrices()
    sm_matrix = sm_matrix[:, sm_matrix.var_names[51:60]]
    sm_matrix = sm_matrix[overlap_matrix.columns]
    sm_matrix.obs_names = PIXEL_PRE + sm_matrix.obs_names

    cell_sm_matrix = cell_sm_matrix[:, sm_matrix.var_names]
    cell_sm_matrix = cell_sm_matrix[overlap_matrix.index]
    
    sm_matrix.write(os.path.join(target_path, 'am_spatiomolecular_adata.h5ad'))
    cell_sm_matrix.write(os.path.join(target_path, 'cell_spatiomolecular_adata.h5ad'))

    return sm_matrix

def correct_dataset(sm_matrix, write = False):
    reload(src.correction)
    from src.correction import get_molecule_normalization_factors, correct_intensities_quantile_regression_parallel, add_matrices

    overlap_matrix, sampling_spec_matrix = get_matrices()
    add_matrices(sm_matrix, overlap_matrix, sampling_spec_matrix)

    total_pixel_overlap, full_pixel_intensities_median = get_molecule_normalization_factors(sm_matrix.to_df(), overlap_matrix, method= st.median)
    
    corr_sm_matrix = correct_intensities_quantile_regression_parallel(sm_matrix, total_pixel_overlap, full_pixel_intensities_median, reference_ions=sm_matrix.var_names, n_jobs=multiprocessing.cpu_count())
    
    if write:
        corr_sm_matrix.write(os.path.join(target_path, 'am_spatiomolecular_adata_corrected.h5ad'))

    return corr_sm_matrix


def deconvolute_dataset(corr_sm_matrix, cell_sm_matrix, write = False):
    reload(src.correction)
    from src.correction import cell_normalization_Rappez_adata
    overlap_matrix, sampling_spec_matrix = get_matrices()

    corr_cell_sm_matrix = cell_normalization_Rappez_adata(sampling_prop_matrix=overlap_matrix, sampling_spec_matrix=sampling_spec_matrix, adata=corr_sm_matrix, raw_adata=cell_sm_matrix)
    
    if write:
        corr_cell_sm_matrix.write(os.path.join(target_path, 'cell_spatiomolecular_adata_corrected.h5ad'))
    
    return corr_cell_sm_matrix

def test_correct_dataset(sm_matrix, corr_sm_matrix):
    gen_corr_sm_matrix = correct_dataset(sm_matrix)

    assert len(corr_sm_matrix.to_df().compare(gen_corr_sm_matrix.to_df())) == 0 and len(gen_corr_sm_matrix.to_df().dropna(how='all')) > 0, 'Error in dataset correction: \n' + str(corr_sm_matrix.to_df().compare(gen_corr_sm_matrix.to_df()))
    

def test_deconvolute_dataset(corr_sm_matrix, cell_sm_matrix, corr_cell_sm_matrix):
    gen_corr_cell_sm_matrix = deconvolute_dataset(corr_sm_matrix, cell_sm_matrix)

    assert len(corr_cell_sm_matrix.to_df().compare(gen_corr_cell_sm_matrix.to_df())) == 0 and len(gen_corr_cell_sm_matrix.to_df().dropna(how='all')) > 0, 'Error in dataset correction: \n' + str(corr_cell_sm_matrix.to_df().compare(gen_corr_cell_sm_matrix.to_df()))
    


# if __name__ == '__main__':
  #   _ = create_small_dataset()
   #  _ = correct_dataset(sm_matrix = get_sm_matrix(), write = True)
   #  _ = deconvolute_dataset(corr_sm_matrix = get_corr_sm_matrix(), cell_sm_matrix = get_cell_sm_matrix(), write = True)
    