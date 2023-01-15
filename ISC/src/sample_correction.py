import os
import json
import pandas as pd
import numpy as np
import scanpy as sc
import statistics as st
from ISC.src import const
from ISC.src import correction as corr

class SampleCorrection:
    
    def __init__(self, sample, config, n_jobs, verbose=True):
        
        self.name = sample
        self.config = config
        self.n_jobs = n_jobs
        self.v = verbose
        
        if self.v: print("Preparing, loading file for corrections for sample %s" % sample)
        
        self.spacem_config = self.get_spacem_config()
        self.analysis_prefix = os.path.join(self.config['runtime']['spacem_dataset_path'], 
                                            self.name)
        
        self.load_data_files()
        
        if self.v: print("Iterating through ions for sample %s" % sample)
        self.correct_suppression()

        if self.v: print("Running deconvolution for sample %s" % sample)
        self.deconvolution()
        
        if self.v: print("Saving sample %s" % sample)
        if self.config['output']['also_write_sample_results']:
            sample_out_folder = os.path.join(self.config['runtime']["out_folder"], 
                                             self.name,
                                             self.config['output']['write_sample_folder_path'])

            if not os.path.exists(sample_out_folder):
                os.makedirs(sample_out_folder)

            self.cell_adata.write(os.path.join(sample_out_folder, self.config['output']['file_names']['adata']))
            self.gen_cell_adata.write(os.path.join(sample_out_folder, self.config['output']['file_names']['generated_adata']))
            self.corr_cell_adata.write(os.path.join(sample_out_folder, self.config['output']['file_names']['corrected_adata']))

            if self.config['output']['save_am_files']:
                self.corr_am_adata.write(os.path.join(sample_out_folder, self.config['output']['file_names']['corrected_am_adata']))
        
    

    def get_spacem_config(self):
        with open(os.path.join(self.config['runtime']['spacem_dataset_path'], 
                                self.name,
                                self.config['input']['spacem_config_file'])) as json_file:
            data = json.load(json_file)
        return data
    
    def load_data_files(self):
        self.cell_regions = pd.read_csv(os.path.join(self.analysis_prefix,
                                                     self.config['input']['cell_regions_file'])
                                       )
        self.mark_regions = pd.read_csv(os.path.join(self.analysis_prefix,
                                                     self.config['input']['mark_regions_file'])
                                       )
        self.overlap_regions = pd.read_csv(os.path.join(self.analysis_prefix,
                                                        self.config['input']['overlap_regions_file'])
                                       )

        self.am_adata = sc.read(os.path.join(self.analysis_prefix,
                                             self.config['input']['am_adata_file'])
                                       )
        self.cell_adata = sc.read(os.path.join(self.analysis_prefix,
                                               self.config['input']['cell_adata_file'])
                                       )
    def correct_suppression(self):
   #    'full_pixel_avg_intensity_method': 'median', 
   #    'correction_proportion_threshold': 0.01, # smaller values remove influence of sampling proportion better, but may suffer more from outliers
   #    'correction_ratios_normalize': False,
   #    'correction_intercept': True,
        
        corr.add_overlap_matrix_spacem(self.am_adata, self.cell_regions, 
                                       self.mark_regions, self.overlap_regions)
        
        if self.config['correction']['full_pixel_avg_intensity_method'] == 'median':
            method = st.median
        elif self.config['correction']['full_pixel_avg_intensity_method'] == 'mean':
            method = st.mean
        else:
            raise NotImplementedError(('Method "%s" is not implemented for full_pixel_avg_intensity_method.' +
                'Use "median" or "mean" instead. Modify your config!')%(
                self.config['correction']['full_pixel_avg_intensity_method']))
        
        corr.add_normalization_factors(adata=self.am_adata, method=st.median)
        
        self.ref_pool = corr.get_reference_pool(self.am_adata, 
            normalized=self.config['correction']['correction_ratios_normalize'])
        
        # perform the actual quantile regression
        self.corr_am_adata = corr.correct_quantile_inplace(adata=self.am_adata,
            reference_ions = self.ref_pool, 
            correct_intersect = self.config['correction']['correction_intercept'],
            normalized = self.config['correction']['correction_ratios_normalize'],
            proportion_threshold=self.config['correction']['correction_proportion_threshold'],
            n_jobs=self.n_jobs,
            progress = self.v)
        
    def deconvolution(self):
        
        overlap_data = corr.get_overlap_data(self.cell_regions, self.mark_regions, self.overlap_regions)
        
        if self.config['deconvolution']['use_data_from_spacem_configuration']:
            deconv_info = self.spacem_config['single_cell_analysis']
        else:
            deconv_info = self.config['deconvolution']
        
        self.corr_cell_adata = corr.deconvolution_spacem(adata=self.corr_am_adata,
            overlap_data=overlap_data,
            raw_adata=self.cell_adata,
            deconvolution_params=deconv_info)
        
        self.gen_cell_adata = corr.deconvolution_spacem(adata=self.am_adata,
            overlap_data=overlap_data,
            raw_adata=self.cell_adata,
            deconvolution_params=deconv_info)

        # hand over TPOs to spatiomolecular matrix for downstream analysis
        min_overlap = deconv_info['ablation_marks_min_overlap_ratio']
        self.corr_cell_adata.obs['list_TPO'] = self.assign_average_tpo(self.am_adata, overlap_data, min_overlap, method=lambda x: ";".join(x.astype(str)))
        self.gen_cell_adata.obs['list_TPO'] = self.assign_average_tpo(self.am_adata, overlap_data, min_overlap, method=lambda x: ";".join(x.astype(str)))

    def assign_average_tpo(self, am_adata, overlap_data, min_overlap, method=np.mean):
        if min_overlap is None:
            min_overlap = 0

        overlap = overlap_data.overlap_regions
        overlap['am_id'] = overlap['am_id'].astype(str)
        overlap['cell_id'] = overlap['cell_id'].astype(str)
        merged_df = pd.merge(overlap[['am_id', 'cell_id']], am_adata.obs[const.TPO], left_on='am_id', right_index=True)
        merged_df = merged_df[merged_df[const.TPO] >= min_overlap]

        mean_df = merged_df[['cell_id', 'correction_total_pixel_overlap']].groupby('cell_id', group_keys=False).agg(method)
        return mean_df[const.TPO]

   