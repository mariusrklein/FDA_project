""" ISC (ion suppression correction)

This is the projects main class that holds all the steps of the correction workflow

Author: Marius Klein (mklein@duck.com), January 2023
"""

import os
import json
import re
import multiprocessing
import functools
from typing import Union
import warnings
import pandas as pd
import numpy as np
import anndata as ad
from joblib import Parallel, delayed
from tqdm import tqdm
from ISC.src.correction_evaluation import CorrectionEvaluation
from ISC.src.sample_correction import SampleCorrection
from ISC.src import const

class IonSuppressionCorrection:
    """IonSuppressionCorrection class

    This is the projects main class that holds all the steps of the correction workflow.

    Methods:
     * __init__: sets up workflow
     * run: actually runs workflow

    """

    corrections = []
    adata = None
    gen_adata = None
    corr_adata = None
    file_locations = {}
    evaluation = None

    def __init__(self,
        source_path: str,
        config: Union[str, dict] = None,
        n_jobs: int = None,
        verbose: bool = True
    ) -> None:
        """prepare data for ion suppression correction

        Args:
            source_path (str): path to SpaceM data folder
            config (Union[str, dict], optional): path to ISC config dictionary or path to 
            corresponding JSON file. Defaults to None.
            n_jobs (int, optional): Number of CPU cores to use. Defaults to None. If not specified,
            takes mutliprocessing.cpu_count()
            verbose (bool, optional): controls amount of output to command line. Defaults to True.
        """

        self.v = verbose

        # ensuring that we have some info on number of available cores for correction/deconvolution
        if n_jobs is not None:
            self.n_jobs=n_jobs
        else:
            self.n_jobs = multiprocessing.cpu_count()

        #
        if config is None:
            self.config = const.CONFIG
        elif isinstance(config, str):
            with open(config, encoding='UTF8') as json_file:
                self.config = json.load(json_file)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise IOError("The config argument has to be a dictionary or a string that specifies "+
                "the location of a JSON config file.")

        self.config['runtime'] = {}
        self.config['runtime']['spacem_dataset_path'] = source_path

        self.check_config()

        self.spacem_metadata = self.get_spacem_metadata()
        self.samples_list = self.get_samples()

    def run(self):
        """AI is creating summary for run
        """
        
        if self.v:
            print("Starting calculations for individual samples.")

        self.corrections = self.init_sample_corrections()
        
        if self.v:
            print("Combining all samples.")
        self.combine_wells()
        
        if self.v:
            print("Writing combined data matrices.")
        self.write_outputs()
        
        if self.v:
            print("Running evaluation notebooks.")
        self.trigger_evaluation()
        
        if self.v:
            print("Done with evaluations")
        
        
    def check_config(self):
        """assert that given configuration works

        Currently, the presence of the spacem_dataset_path (input directory) and the two output 
        paths (corrected data, evaluation files) is checked.

        Raises:
            FileNotFoundError: Raised when a specified directory does not exist.
            OSError: Raised when results should be safed to input folder but the location is not 
            writable.
            Warning: When internal output folder already exists. Risks that files get overwritten.
        """
        

        if not os.path.isdir(self.config['runtime']['spacem_dataset_path']):
            raise FileNotFoundError("Supplied directory with SpaceM data was not found.")
            
        def check_out_directory(path_label, path_name):
            if self.config['output']['write_to_input_folder']:
                self.config['runtime'][path_label] = os.path.join(
                    self.config['runtime']['spacem_dataset_path'],
                    path_name
                )
            else:
                if self.config['output']['external_output_folder'] is None:
                    raise FileNotFoundError("Forbidden to write to input folder and no directory " +
                        "supplied. Modify your config!")

                self.config['runtime'][path_label] = os.path.join(
                    self.config['output']['external_output_folder'],
                    path_name
                )

            if os.path.isdir(self.config['runtime'][path_label]):
                warnings.warn(f"Output directory {self.config['runtime'][path_label]} exists. " +
                    "Contained files may be overwritten.")
            else:
                try:
                    os.makedirs(self.config['runtime'][path_label])
                except Exception as exc:
                    raise OSError(f"The output directory {self.config['runtime'][path_label]} "+
                        "could not be created. Modify your config!") from exc

        check_out_directory(path_label = "out_folder",
                            path_name = self.config['output']['results_folder'])
        check_out_directory(path_label = "evaluation_folder",
                            path_name = self.config['evaluation']['evaluation_folder'])
        
        
    def get_spacem_metadata(self) -> pd.DataFrame:
        """ loads and prepares the spacem metadata file which contains all the processed wells

        Raises:
            FileNotFoundError: Raiseed when metadata csv file can't be found
            KeyError: Raised when column pattern from ISC config does not match given metadata csv

        Returns:
            pd.DataFrame: Table with rows representing wells that were processed by SpaceM.
        """
        metadata_file = os.path.join(self.config['runtime']['spacem_dataset_path'],
                                     self.config['input']['spacem_dataset_metadata_file'])
        try:
            spacem_metadata = pd.read_csv(metadata_file)
            
            # we want a globally defined sample/well column, creating this using pattern from config
            spacem_metadata[const.SAMPLE_COL] = spacem_metadata.agg(
                re.sub('{(.*?)}',
                    r'{0[\1]}',
                    self.config['input']['spacem_dataset_metadata_well_name']).format,
                axis=1
            )

        except FileNotFoundError as exc:
            raise FileNotFoundError(f"No spacem_dataset_metadata_file found at {metadata_file}. " +
                "Modify your config!") from exc
        
        except KeyError as exc:
            raise KeyError(f"""The spacem_dataset_metadata_file at {metadata_file} does not contain 
columns to create well name using spacem_dataset_metadata_well_name 
{self.config['input']['spacem_dataset_metadata_well_name']}. Modify your config!""") from exc
        
        # same as well column we globally define a population column for differen conditions etc.
        try:
            spacem_metadata[const.POPULATION_COL] = spacem_metadata.agg(
                re.sub('{(.*?)}', 
                    r'{0[\1]}', 
                    self.config['input']['spacem_dataset_metadata_population_name']).format, 
                axis=1
            )

        except KeyError as exc:
            raise KeyError(f"""The spacem_dataset_metadata_file at {metadata_file} does not contain 
columns to create well name using spacem_dataset_metadata_well_name 
{self.config['input']['spacem_dataset_metadata_population_name']}. Modify your config!""") from exc

        # if the right column patterns were given, the resulting data frame can be returned
        return spacem_metadata
    

    def get_samples(self,
        path: str = None
    ) -> list:
        """collects samples/wells to be processed in correction from spacem_dataset_path and
        metadata file. Samples are only included that are present in both resources

        Args:
            path (str, optional):  path to the spacem data folder that contains samples (subfolders)
            Defaults to None (then getting spacem_dataset_path from config)

        Returns:
            list: list of str that contain names of included samples.
        """

        if path is None:
            path = self.config['runtime']['spacem_dataset_path']
            
        samples = []
        for dirpath, dirnames, _ in os.walk(path):
            if 'analysis' in dirnames:
                samples.append(re.sub(path+'/?', '', dirpath))
        
        if self.v:
            print(f"Found {len(samples)} samples in given folder: {', '.join(samples)}")

        samples_list = [s for s in samples
            if s in list(self.spacem_metadata[const.SAMPLE_COL])]
        samples_excluded = [s for s in samples
            if s not in list(self.spacem_metadata[const.SAMPLE_COL])]
        if len(samples_excluded) > 0:
            if self.v:
                print(f"Excluded {len(samples_excluded)} samples as they did not exist in "+
                    f"{self.config['input']['spacem_dataset_metadata_file']}: " +
                    f"{', '.join(samples_excluded)}")
                print(f"Proceeding with samples {', '.join(samples_list)}")
        
        self.config['runtime']['samples'] = samples_list

        return samples_list
    
    
    def init_sample_corrections(self, samples: list[str] = None) -> list[SampleCorrection]:
        """ run corrections and deconvolutions of samples

        Args:
            samples (list[str], optional): list of sample names as provided by get_samples()

        Returns:
            list[SampleCorrection]: list of SampleCorrection objects which hold all the information
            on the corrected and deconvoluted samples.
        """
        
        if samples is None:
            samples = self.samples_list

        jobs_samples = np.min([self.n_jobs, len(self.samples_list)])
        jobs_corr = 1
            
        if self.v:
            print(f"using {jobs_samples} times {jobs_corr} cores for calculations.")
        
        if jobs_samples > 1:
            corrections = Parallel(n_jobs=jobs_samples)(
                delayed(SampleCorrection)(sample=sample,
                                          config=self.config,
                                          n_jobs=jobs_corr,
                                          verbose=False) for sample in tqdm(samples))
        else:
            corrections = [SampleCorrection(sample=sample,
                                          config=self.config,
                                          n_jobs=jobs_corr,
                                          verbose=self.v) for sample in tqdm(samples)]
            

        return corrections
    
    
    def combine_wells(self, sample_list: list[SampleCorrection] = None):
        """combine corrected and deconvoluted wells to a common AnnData Set

        Args:
            sample_list (list[SampleCorrection], optional): List of corrected and deconvoluted
            samples. Defaults to None (then taking samples from self.)
        """
        adatas = {}
        gen_adatas = {}
        corr_adatas = {}

        if sample_list is None:
            sample_list = self.corrections
        
        for sample in sample_list:
            adatas[sample.name] = sample.cell_adata
            gen_adatas[sample.name] = sample.gen_cell_adata
            corr_adatas[sample.name] = sample.corr_cell_adata
            
        self.adata = self.concat_wells(adatas)
        self.gen_adata = self.concat_wells(gen_adatas)
        self.corr_adata = self.concat_wells(corr_adatas)
            
    
    def concat_wells(self, adata_dict: dict[str, ad.AnnData]) -> ad.AnnData:
        """helper that combines adatas of multiple samples to one. Certain columns of AnnData.var
        are aggregated using average of sum functions 

        Args:
            adata_dict (dict[ad.AnnData]): dictionary that contains anndatas for concatenation as
            values and corresponding sample names as keys.

        Returns:
            ad.AnnData: combined adata.
        """
        
        # AnnData.var columns that should be aggregated using mean function
        averaged_cols = ['correction_full_pixel_avg_intensities',
                                    'correction_quantreg_slope',
                                    'correction_quantreg_intersect']

        # AnnData.var columns that should be aggregated using sum function
        counted_cols = ['correction_using_ion_pool']

        # TODO figure out whether to use outer or inner join
        adata = ad.concat(adata_dict,
            label=const.SAMPLE_COL,
            index_unique="_",
            merge="first",
            join='inner',
            fill_value=0)

        # concatenating var tables
        concatenated_var_df = pd.concat(
            {
                k: v.var for k, v in adata_dict.items()
            }).select_dtypes(include=[float, bool])

        # calculating mean/sum of correction columns only if correction was performed on dataset
        if 'correction_quantreg_slope' in concatenated_var_df.columns:
            # process columns to be averaged
            mean_var_df = concatenated_var_df.reset_index(
                names = [const.SAMPLE_COL, 'ion']
                ).groupby('ion')[averaged_cols].mean(numeric_only = True)

            mean_var_df.columns = ['mean_'+col for col in mean_var_df.columns]

            # process standard deviation of columns to be averaged
            std_var_df = concatenated_var_df.reset_index(
                names = [const.SAMPLE_COL, 'ion']
                ).groupby('ion')[averaged_cols].std(numeric_only = True)

            std_var_df.columns = ['sd_'+col for col in std_var_df.columns]

            # process columns to be counted
            count_var_df = concatenated_var_df.reset_index(
                names = [const.SAMPLE_COL, 'ion']
                ).groupby('ion')[counted_cols].sum(numeric_only = True)

            count_var_df.columns = ['sum_'+col for col in count_var_df.columns]

            dfs = [adata.var, mean_var_df, std_var_df, count_var_df]
            
            # merge can only take two tables, this command merges all four sequentially
            adata.var = functools.reduce(lambda left,right: pd.merge(
                left,
                right,
                how='left',
                left_index=True,
                right_index=True
            ), dfs)

            adata.var['corrected_only_using_pool'] = (
                adata.var['sum_correction_using_ion_pool'] == len(adata_dict))

        # remove original columns in automatically concatenated AnnData.var table, as they reflect
        # only the first sample 
        for col in adata.var.columns:
            if col in ['correction_full_pixel_avg_intensities', 'correction_n_datapoints',
                       'correction_n_iterations', 'correction_quantreg_intersect',
                       'correction_quantreg_slope', 'correction_using_ion_pool']:
                del adata.var[col]


        return adata
    
    
    def write_outputs(self):
        """save generated anndatas to file
        """

        # building file paths to save to
        self.file_locations = {
            k: os.path.join(self.config['runtime']["out_folder"], v)
            for k, v in self.config['output']['file_names'].items()
        }

        self.config['runtime']['file_locations'] = self.file_locations
        
        # actually saving files
        self.adata.write(self.file_locations['adata'])
        self.gen_adata.write(self.file_locations['generated_adata'])
        self.corr_adata.write(self.file_locations['corrected_adata'])

   
    def trigger_evaluation(self):
        """check config specification if evaluation notebooks should be run and trigger them 
        """
        
        # skip evaluation if none are requested
        if (not self.config['evaluation']['run_qc']
                and not self.config['evaluation']['run_results_evaluation']
                and not self.config['evaluation']['run_features_evaluation']):
            return
        
        # initialize evaluation object
        self.evaluation = CorrectionEvaluation(correction=self)
        
    
    def save_config(self, file: str):
        """save current config to JSON file.

        Args:
            file (str): file path and name to save to.
        """
        with open(file, mode="w", encoding="UTF8") as fp:
            json.dump(self.config , fp)
