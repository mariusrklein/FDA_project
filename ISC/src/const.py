# CONSTANTS

CELL_PRE = ''
PIXEL_PRE = ''
TPO = 'correction_total_pixel_overlap'
FPAI = 'correction_full_pixel_avg_intensities'
SAMPLE_COL = "sample_col"
POPULATION_COL = "population_col"

CONFIG = {
    'input': {
        'spacem_config_file': 'config.json',
        'spacem_dataset_metadata_file': 'metadata.csv',
        'spacem_dataset_metadata_well_name': '{row}{col}',
        'spacem_dataset_metadata_population_name': '{treatment}',
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
        'correction_ratios_normalize': False,
    },
    'deconvolution': {
        'use_data_from_spacem_configuration': True, # If True, wells may be treated differently as every well has their own config file
        'cell_normalization_method': 'weighted_by_overlap_and_sampling_area', # disregarded if use_data_from_spacem_configuration=True
        'ablation_marks_min_overlap_ratio': 0.3, # disregarded if use_data_from_spacem_configuration=True
    },
    'evaluation': {
        'evaluation_folder': 'ion_suppression_correction/evaluation',
        'run_qc': True,
        'run_results_evaluation': True,
    },
    'output': {
        'save_am_files': True,
        'results_folder': 'ion_suppression_correction/output',
        'write_to_input_folder': True, # True writes sample-based results into sample folders and global results to input_folder_path
        'also_write_sample_results': True,
        'write_sample_folder_path': 'ion_suppression_correction',
        'external_output_folder': None,
        'file_names': {
            'corrected_am_adata': "corrected_am_sm_matrix.h5ad",
            'adata': "original_batch_sm_matrix.h5ad",
            'generated_adata': "gen_batch_sm_matrix.h5ad",
            'corrected_adata': "corrected_batch_sm_matrix.h5ad"
        },
    },
}

LABEL = {
    'SProp': r'sampling proportion $p_s$',
    'IRatio': r'intensity proportion ratio $\mu$',
    'RDens': r'relative deviation $\frac{J_{Python} - J_R}{J_{Python}}$',
    'RDev_general': r'relative deviation $\frac{J_{2} - J_1}{J_{1}}$'
}

