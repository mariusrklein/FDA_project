import os
import papermill as pm

notebooks_location = "/home/mklein/FDA_project/src"

class CorrectionEvaluation:
    
    def __init__(self, correction):
        
        # this is an instance of ISC
        self.data = correction
        self.config = self.data.config
        
        if self.config['evaluation']['run_qc']:
            self.run_qc()
            
        if self.config['evaluation']['run_feature_evaluation']:
            self.run_feature_eval()
        
        if self.config['evaluation']['run_results_evaluation']:
            self.run_performance_eval()
        
        
    def run_qc(self):
        pm.execute_notebook(
            os.path.join(notebooks_location, 'notebook_templates', 'qc.ipynb'),
            os.path.join(self.config['runtime']['evaluation_folder'], 'qc.ipynb'),
            parameters={'config': self.config}
        )
    
    def run_feature_eval(self):
        pm.execute_notebook(
            os.path.join(notebooks_location, 'notebook_templates', 'feature_analysis.ipynb'),
            os.path.join(self.config['runtime']['evaluation_folder'], 'feature_analysis.ipynb'),
            parameters={'config': self.config}
        )
    
    def run_performance_eval(self):
        pm.execute_notebook(
            os.path.join(notebooks_location, 'notebook_templates', 'results_analysis.ipynb'),
            os.path.join(self.config['runtime']['evaluation_folder'], 'results_analysis.ipynb'),
            parameters={'config': self.config}
        )