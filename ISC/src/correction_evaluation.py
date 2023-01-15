import os
import papermill as pm
from importlib_resources import files, as_file

class CorrectionEvaluation:
    
    def __init__(self, correction):
        
        # this is an instance of ISC
        self.data = correction
        self.config = self.data.config
        
        self.find_notebooks()
        
        
        if self.config['evaluation']['run_qc']:
            self.run_qc()
        
        if self.config['evaluation']['run_results_evaluation']:
            self.run_performance_eval()
        
    def find_notebooks(self):
        qc = files('src.notebook_templates').joinpath('qc.ipynb')
        ra = files('src.notebook_templates').joinpath('results_analysis.ipynb')

        with as_file(qc) as path:
            self.qc_path = path

        with as_file(ra) as path:
            self.ra_path = path
            
        
        
    def run_qc(self):
        pm.execute_notebook(self.qc_path,
            os.path.join(self.config['runtime']['evaluation_folder'], 'qc.ipynb'),
            parameters={'config': self.config, 'package_path': }
        )
    
    def run_performance_eval(self):
        pm.execute_notebook(self.ra_path,
            os.path.join(self.config['runtime']['evaluation_folder'], 'results_analysis.ipynb'),
            parameters={'config': self.config}
        )
        
        
if __name__ == '__main__':
    qc = files('src.notebook_templates').joinpath('qc.ipynb')
    ra = files('src.notebook_templates').joinpath('results_analysis.ipynb')

    with as_file(qc) as path:
        print(path)

    with as_file(ra) as path:
        print(path)