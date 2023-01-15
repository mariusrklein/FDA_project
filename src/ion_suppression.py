import os
import sys
import json
import re
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import statistics as st
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import functools
import papermill as pm
from src.correction_evaluation import CorrectionEvaluation
from src.sample_correction import SampleCorrection
from src import const

class ISC:
    
    def __init__(self, source_path, config = None, n_jobs = None, verbose=True):
        
        self.v = verbose
        
        if n_jobs is not None:
            self.n_jobs=n_jobs
        else:
            self.n_jobs = multiprocessing.cpu_count()
        
        if config is None:
            self.config = const.CONFIG
        elif type(config) is str:
            with open(config) as json_file:
                self.config = json.load(json_file)
        else:
            self.config = config
            
        self.config['runtime'] = {}
        self.config['runtime']['spacem_dataset_path'] = source_path
        
        self.check_config()
        
        self.spacem_metadata = self.get_spacem_metadata()
        self.samples_list = self.get_samples() 
        
    def run(self):
        
        if self.v: print("Starting calculations for individual samples.")
        self.corrections = self.init_sample_corrections()
        
        if self.v: print("Combining all samples.")
        self.combine_wells()
        
        if self.v: print("Writing combined data matrices.")
        self.write_outputs()
        
        if self.v: print("Running evaluation notebooks.")
        self.trigger_evaluation()
        
        print("DONE!")
        
        
    def check_config(self):
        
        if not os.path.isdir(self.config['runtime']['spacem_dataset_path']):
            raise FileNotFoundError("Supplied directory with SpaceM data was not found.")
            
        def check_out_directory(path_label, path_name):
            if self.config['output']['write_to_input_folder']:
                self.config['runtime'][path_label] = os.path.join(self.config['runtime']['spacem_dataset_path'], path_name)
            else:
                if self.config['output']['external_output_folder'] is None:
                    raise FileNotFoundError("Forbidden to write to input folder and no directory supplied. Modify your config!")

                self.config['runtime'][path_label] = os.path.join(self.config['output']['external_output_folder'], path_name)

            if os.path.isdir(self.config['runtime'][path_label]):
                print('Output directory exists. Contained files may be overwritten.')
       #          if os.listdir(self.config['runtime'][path_label]):
       #              raise FileExistsError(('The supplied output folder "%s" already exists and is not empty. '+
       #                                     'Refusing to write to this folder. Modify your config!')%self.config['runtime'][path_label])
            else:
                try:
                    os.makedirs(self.config['runtime'][path_label])
                except:
                    raise Exception('The output directory "%s" could not be created. Modify your config!'%self.config['runtime'][path_label])

        check_out_directory(path_label = "out_folder", path_name = self.config['output']['results_folder'])
        check_out_directory(path_label = "evaluation_folder", path_name = self.config['evaluation']['evaluation_folder'])
        
        
    def get_spacem_metadata(self):
        
        metadata_file = os.path.join(self.config['runtime']['spacem_dataset_path'], 
                                     self.config['input']['spacem_dataset_metadata_file'])
        try:
            spacem_metadata = pd.read_csv(metadata_file)
            spacem_metadata[const.SAMPLE_COL] = spacem_metadata.agg(
                re.sub('{(.*?)}', r'{0[\1]}', self.config['input']['spacem_dataset_metadata_well_name']).format, 
                axis=1)
        except FileNotFoundError:
            raise FileNotFoundError('No spacem_dataset_metadata_file found at %s. Modify your config!'%(
                metadata_file))
        except KeyError:
            raise KeyError('''The spacem_dataset_metadata_file at %s does not contain columns 
                to create well name using spacem_dataset_metadata_well_name %s. Modify your config!'''%(
                metadata_file, self.config['input']['spacem_dataset_metadata_well_name']))
        try:
            spacem_metadata[const.POPULATION_COL] = spacem_metadata.agg(
                re.sub('{(.*?)}', r'{0[\1]}', self.config['input']['spacem_dataset_metadata_population_name']).format, 
                axis=1)
        except KeyError:
            raise KeyError('''The spacem_dataset_metadata_file at %s does not contain columns 
                to create well name using spacem_dataset_metadata_population_name %s. Modify your config!'''%(
                metadata_file, self.config['input']['spacem_dataset_metadata_population_name']))
        return spacem_metadata
    
    def get_samples(self, path = None):
        
        if path is None:
            path = self.config['runtime']['spacem_dataset_path']
            
        samples = []
        for dirpath, dirnames, filenames in os.walk(path):
            if 'analysis' in dirnames:
                samples.append(re.sub(path+'/?', '', dirpath))
        
        if self.v: print("Found %d samples in given folder: %s"%(len(samples), ", ".join(samples)))
        samples_list = [s for s in samples if s in list(self.spacem_metadata[const.SAMPLE_COL])]
        samples_excluded = [s for s in samples if s not in list(self.spacem_metadata[const.SAMPLE_COL])]
        if len(samples_excluded) > 0:
            if self.v: print("Excluded %d samples as they did not exist in %s: %s"%(
                             len(samples_excluded), 
                             self.config['input']['spacem_dataset_metadata_file'],
                             ", ".join(samples_excluded)))
        
        self.config['runtime']['samples'] = samples_list
        
        return samples_list
    
    
    def init_sample_corrections(self):
        
        
        jobs_samples = np.min([self.n_jobs, len(self.samples_list)])
        jobs_corr = 1
            
        if self.v: print("using %d times %d cores for calculations."%(jobs_samples, jobs_corr))
        
        if jobs_samples > 1:
            corrections = Parallel(n_jobs=jobs_samples)(
                delayed(SampleCorrection)(sample=sample,
                                          config=self.config, 
                                          n_jobs=jobs_corr,
                                          verbose=False) for sample in tqdm(self.samples_list))
        else:
            corrections = [SampleCorrection(sample=sample,
                                          config=self.config, 
                                          n_jobs=jobs_corr,
                                          verbose=self.v) for sample in tqdm(self.samples_list)]
            

        return corrections
    
    
    def combine_wells(self):
        
        adatas = {}
        gen_adatas = {}
        corr_adatas = {}
        
        for sample in self.corrections:
            adatas[sample.name] = sample.cell_adata
            gen_adatas[sample.name] = sample.gen_cell_adata
            corr_adatas[sample.name] = sample.corr_cell_adata
            
        self.adata = self.concat_wells(adatas)
        self.gen_adata = self.concat_wells(gen_adatas)
        self.corr_adata = self.concat_wells(corr_adatas)
            
    
    def concat_wells(self, adata_dict):
    
        averaged_cols = ['correction_full_pixel_avg_intensities', 'correction_quantreg_slope', 'correction_quantreg_intersect']
        counted_cols = ['correction_using_ion_pool']

        adata = ad.concat(adata_dict, label=const.SAMPLE_COL, index_unique="_", merge="first", join='inner', fill_value=0)
        concatenated_var_df = pd.concat({k: v.var for k, v in adata_dict.items()}).select_dtypes(include=[float, bool])

        if 'correction_quantreg_slope' in concatenated_var_df.columns:
            mean_var_df = concatenated_var_df.reset_index(names = [const.SAMPLE_COL, 'ion']).groupby('ion')[averaged_cols].mean(numeric_only = True)
            mean_var_df.columns = ['mean_'+col for col in mean_var_df.columns]

            std_var_df = concatenated_var_df.reset_index(names = [const.SAMPLE_COL, 'ion']).groupby('ion')[averaged_cols].std(numeric_only = True)
            std_var_df.columns = ['sd_'+col for col in std_var_df.columns]

            count_var_df = concatenated_var_df.reset_index(names = [const.SAMPLE_COL, 'ion']).groupby('ion')[counted_cols].sum(numeric_only = True)
            count_var_df.columns = ['sum_'+col for col in count_var_df.columns]

            dfs = [adata.var, mean_var_df, std_var_df, count_var_df]
            adata.var = functools.reduce(lambda left,right: pd.merge(left, right, how='left', left_index=True, right_index=True), dfs)
            adata.var['corrected_only_using_pool'] = adata.var['sum_correction_using_ion_pool'] == len(adata_dict)

        for col in adata.var.columns:
            if col in ['correction_full_pixel_avg_intensities', 'correction_n_datapoints', 'correction_n_iterations', 
                       'correction_quantreg_intersect', 'correction_quantreg_slope', 'correction_using_ion_pool']:
                del adata.var[col]

        #sc.tl.pca(adata)
        #sc.external.pp.bbknn(adata, batch_key=const.SAMPLE_COL)
        return adata
    
    
    def write_outputs(self):

        self.file_locations = {k: os.path.join(self.config['runtime']["out_folder"], v) for k, v in self.config['output']['file_names'].items()}
        self.config['runtime']['file_locations'] = self.file_locations
        
        self.adata.write(self.file_locations['adata'])
        self.gen_adata.write(self.file_locations['generated_adata'])
        self.corr_adata.write(self.file_locations['corrected_adata'])

   
    def trigger_evaluation(self):
 #       'evaluation': {
 #      'evaluation_folder': 'ion_suppression_correction/evaluation',
 #      'run_qc': True,
 #      'run_results_evaluation': True,
 #      'run_features_evaluation': True,
 #  },
        
        if (not self.config['evaluation']['run_qc'] 
                and not self.config['evaluation']['run_results_evaluation'] 
                and not self.config['evaluation']['run_features_evaluation']): 
            return
        
        self.evaluation = CorrectionEvaluation(correction=self)
        
    
    def save_config(self, file):
        with open(file, "w") as fp:
            json.dump(self.config , fp) 
    
    
if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) < 1 or len(args) < 1:
        raise IOError('Please give 1 to 2 arguments, separated by space: '
                      +'Directory of the SpaceM-generated folder structure and '
                      +'directory/name of the configuration file to be used.')


    print('Running ion suppression correction with folder "%s"'%args[0])
    if len(args) == 2:
        print('Using config file at "%s"'%args[1])
        conf = args[1]
    else:
        print('Using default config:')
        print(const.CONFIG)
        conf=None
        
    isc = ISC(args[0], conf)
    isc.run()
        
    
    