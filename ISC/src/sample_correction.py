""" SampleCorrection

this class handles the specific correction and deconvolution of individual samples/wells.

Author: Marius Klein (mklein@duck.com), January 2023
"""

import os
import json
import statistics as st
import pandas as pd
import anndata as ad
import scanpy as sc
from SpaceM.lib.modules.overlap_analysis import OverlapData
from ISC.src import const
from ISC.src import correction as corr

class SampleCorrection:
    """SampleCorrection

    this class handles the specific correction and deconvolution of individual samples/wells.

    methods:
     * __init__: runs all steps on a sample: correction and deconvolution

    """
    
    def __init__(
        self,
        sample: str,
        config: dict[str, any],
        n_jobs: int,
        verbose: bool =True
    ):
        
        self.name = sample
        self.config = config
        self.n_jobs = n_jobs
        self.v = verbose
        
        if self.v:
            print(f"Preparing, loading file for corrections for sample {sample}")
        
        self.spacem_config = self.get_spacem_config()
        self.analysis_prefix = os.path.join(self.config['runtime']['spacem_dataset_path'],
                                                 self.name)
        
        self.load_data_files()
        
        if self.v:
            print(f"Iterating through ions for sample {sample}")
        self.correct_suppression()

        if self.v:
            print(f"Running deconvolution for sample {sample}")
        self.deconvolution()
        
        if self.v:
            print(f"Saving sample {sample}")

        # writing anndatas of individual samples only if specified in config
        if self.config['output']['also_write_sample_results']:

            # constructing save_to folder
            if self.config['output']["write_to_input_folder"]:
                samples_folder = self.config['runtime']["spacem_dataset_path"]
            else:
                samples_folder = self.config['runtime']["out_folder"]
                
            sample_out_folder = os.path.join(
                samples_folder,
                self.name,
                self.config['output']['write_sample_folder_path']
            )

            if not os.path.exists(sample_out_folder):
                os.makedirs(sample_out_folder)

            # saving individual anndatas
            self.cell_adata.write(
                os.path.join(sample_out_folder,
                             self.config['output']['file_names']['adata']
                )
            )
            self.gen_cell_adata.write(
                os.path.join(sample_out_folder,
                             self.config['output']['file_names']['generated_adata']
                )
            )
            self.corr_cell_adata.write(
                os.path.join(sample_out_folder,
                             self.config['output']['file_names']['corrected_adata']
                )
            )

            if self.config['output']['save_am_files']:
                self.corr_am_adata.write(
                    os.path.join(sample_out_folder,
                                 self.config['output']['file_names']['corrected_am_adata']
                    )
                )
        
    

    def get_spacem_config(self) -> dict:
        """loading sample-specific spacem config

        Returns:
            dict: config-dictionary
        """
        
        # constructing sample-specific spacem config file location
        spacem_config_file = os.path.join(
            self.config['runtime']['spacem_dataset_path'],
            self.name,
            self.config['input']['spacem_config_file']
        )

        #loading json file
        with open(spacem_config_file, encoding="UTF8") as json_file:
            data = json.load(json_file)

        return data
    

    def load_data_files(self):
        """loading all required files to correct and deconvolute a sample
        """

        self.cell_regions = pd.read_csv(
            os.path.join(
                self.analysis_prefix,
                self.config['input']['cell_regions_file']
            )
        )

        self.mark_regions = pd.read_csv(
            os.path.join(
                self.analysis_prefix,
                self.config['input']['mark_regions_file']
            )
        )

        self.overlap_regions = pd.read_csv(
            os.path.join(
                self.analysis_prefix,
                self.config['input']['overlap_regions_file']
            )
        )

        self.am_adata = sc.read(
            os.path.join(
                self.analysis_prefix,
                self.config['input']['am_adata_file']
            )
        )

        self.cell_adata = sc.read(
            os.path.join(
                self.analysis_prefix,
                self.config['input']['cell_adata_file']
            )
        )


    def correct_suppression(self):
        """ performs ion suppression correction on individual sample
        """
  
        
        # first construct overlap matrix between ablation marks and cells.
        corr.add_overlap_matrix_spacem(
            adata = self.am_adata,
            cell_regions = self.cell_regions,
            mark_regions = self.mark_regions,
            overlap_regions = self.overlap_regions
        )
        
        if self.config['correction']['full_pixel_avg_intensity_method'] == 'median':
            method = st.median
        elif self.config['correction']['full_pixel_avg_intensity_method'] == 'mean':
            method = st.mean
        else:
            raise NotImplementedError("Method "+
                f"{self.config['correction']['full_pixel_avg_intensity_method']} is not "+
                "implemented for full_pixel_avg_intensity_method. Use 'median' or 'mean' "+
                "instead. Modify your config!")
        

        corr.add_normalization_factors(adata=self.am_adata, method=method)
        
        self.ref_pool = corr.get_reference_pool(self.am_adata,
            normalized=self.config['correction']['correction_ratios_normalize'])
        
        # if no correction is performed, just copy uncorrected anndata
        if self.config['correction']['perform_correction'] is False:
            self.corr_am_adata = self.am_adata
        else:
            # perform the actual quantile regression
            self.corr_am_adata = corr.correct_quantile_inplace(adata=self.am_adata,
                reference_ions = self.ref_pool,
                correct_intersect = self.config['correction']['correction_intercept'],
                normalized = self.config['correction']['correction_ratios_normalize'],
                proportion_threshold=self.config['correction']['correction_proportion_threshold'],
                n_jobs=self.n_jobs,
                progress = self.v)
        
    def deconvolution(self):
        """deconvolution of individual sample
        """
        
        overlap_data = corr.get_overlap_data(
            self.cell_regions, 
            self.mark_regions, 
            self.overlap_regions
        )
        
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
        self.corr_cell_adata.obs['list_TPO'] = self.assign_average_tpo(
            self.am_adata,
            overlap_data,
            min_overlap,
            method=lambda x: ";".join(x.astype(str))
        )

        self.gen_cell_adata.obs['list_TPO'] = self.assign_average_tpo(
            self.am_adata,
            overlap_data,
            min_overlap,
            method=lambda x: ";".join(x.astype(str))
        )

    def assign_average_tpo(
        self,
        am_adata: ad.AnnData,
        overlap_data: OverlapData,
        min_overlap: float,
        method = lambda x: ";".join(x.astype(str))
    ) -> pd.Series:
        """summarise cellular overlap data (total pixel overlap TPO) from am adata to hand over to 
        cell adata

        Args:
            am_adata (ad.AnnData): annotated datamatrix on ablation mark level with information on 
            total pixel overlaps
            overlap_data (OverlapData): SpaceM dataclass that holds spatial information on ablation
            marks, cells and overlaps.
            min_overlap (float): threshold for cellular overlap of ablation marks (only marks with
            overlap above threshold are used in deconvolution)
            method (Callable, optional): used method to aggregate TPO of individual marks that are
            combined to a cell. Defaults to string concatenation.

        Returns:
            pd.Series: Series to be added to cell_adata.obs table
        """
        if min_overlap is None:
            min_overlap = 0

        overlap = overlap_data.overlap_regions
        overlap['am_id'] = overlap['am_id'].astype(str)
        overlap['cell_id'] = overlap['cell_id'].astype(str)
        merged_df = pd.merge(
            overlap[['am_id', 'cell_id']],
            am_adata.obs[const.TPO],
            left_on='am_id',
            right_index=True
        )
        merged_df = merged_df[merged_df[const.TPO] >= min_overlap]

        mean_df = merged_df[['cell_id', const.TPO]].groupby(
            'cell_id', group_keys=False
        ).agg(method)

        return mean_df[const.TPO]

   