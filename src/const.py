# CONSTANTS

CELL_PRE = ''
PIXEL_PRE = ''
TPO = 'correction_total_pixel_overlap'
FPAI = 'correction_full_pixel_avg_intensities'

CONFIG = {
    'input': {
        'spacem_dataset_path': None, # e.g. '/path/to/220412_Luisa_ScSeahorse_SpaceM',
        'spacem_config_file': 'config.json',
        'am_adata_file': 'analysis/ablation_mark_analysis/spatiomolecular_adata.h5ad',
        'cell_adata_file': 'analysis/single_cell_analysis/spatiomolecular_adata.h5ad',
        'overlap_regions_file': 'analysis/overlap_analysis2/overlap.regions.csv',
        'mark_regions_file': 'analysis/overlap_analysis2/ablation_mark.regions.csv',
        'cell_regions_file': 'analysis/overlap_analysis2/cell.regions.csv',
    },
    'correction': {
        'spacem_library_location': None,
        'full_pixel_avg_intensity_method': 'median', 
        'correction_proportion_threshold': 0.1, # smaller values remove influence of sampling proportion better, but may suffer more from outliers
        'correction_intercept': True,
    },
    'deconvolution': {
        'cell_normalization_method': 'weighted_by_overlap_and_sampling_area', # write None if method from spacem config file should be used
        'ablation_marks_min_overlap_ratio': 0.3, # write None if threshold from spacem config file should be used
    },
    'evaluation': {
        'run_qc': True,
        'run_results_evaluation': True,
        'run_features_evaluation': True,
    },
    'ouput': {
        'write_to_input_folder': False,
        'write_folder': None,
        'write_well_am_adata': 'ion_suppression_correction/ablation_mark_analysis/spatiomolecular_adata.h5ad',
        'write_well_cell_adata': 'ion_suppression_correction/single_cell_analysis/spatiomolecular_adata.h5ad',
        'write_dataset_path': 'ion_suppression_correction',
    },
}