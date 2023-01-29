"""
CorrectionEvaluation

This is a project-specific class to handle the creation of evaluation notebooks

Author: Marius Klein (mklein@duck.com), January 2023
"""
import os
import papermill as pm
from importlib_resources import files, as_file

class CorrectionEvaluation:
    """Correction evaluation

    methods:
     * __init__: runs whole evaluation based on given IonSuppressionCorrection data and config.

    """
    
    def __init__(self, correction):
        """trigger whole evaluation process.

        Args:
            correction (IonSuppressionCorrection): Workflow object that contains all info on 
            dataset, only config is relevant at this point.
        """
        # this is an instance of ISC
        self.data = correction
        self.config = self.data.config
        
        # locate templates to insert data into
        self.find_notebooks()
        
        
        if self.config['evaluation']['run_qc']:
            self.run_qc()
        
        if self.config['evaluation']['run_results_evaluation']:
            self.run_performance_eval()
        
    def find_notebooks(self):
        """locate template notebooks for QC and results analysis in package folder.
        """
        qc_file = files('src.notebook_templates').joinpath('qc.ipynb')
        ra_file = files('src.notebook_templates').joinpath('results_analysis.ipynb')

        with as_file(qc_file) as qc_path:
            self.qc_path = qc_path

        with as_file(ra_file) as ra_path:
            self.ra_path = ra_path
            
        
        
    def run_qc(self):
        """Run qc notebook with papermill
        """
        pm.execute_notebook(self.qc_path,
            os.path.join(self.config['runtime']['evaluation_folder'], 'qc.ipynb'),
            parameters={'config': self.config} # TODO: check if notebooks find installation of ISC
        )
    
    def run_performance_eval(self):
        """Run results analysis notebook with papermill
        """
        pm.execute_notebook(self.ra_path,
            os.path.join(self.config['runtime']['evaluation_folder'], 'results_analysis.ipynb'),
            parameters={'config': self.config}
        )



# print locations of notebooks for testing purposes.
if __name__ == '__main__':
    qc = files('src.notebook_templates').joinpath('qc.ipynb')
    ra = files('src.notebook_templates').joinpath('results_analysis.ipynb')

    with as_file(qc) as path:
        print(path)

    with as_file(ra) as path:
        print(path)