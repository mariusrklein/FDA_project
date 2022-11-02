import os 
import papermill as pm


parameters = {
        'source_path': '/g/alexandr/smenon/2022-07-13_Glioblastoma/processed_files',
        'target_path': '/home/mklein/FDA_project/data/Lx_Glioblastoma',
        'condition_name': 'dataset_3'
    }

save_to = parameters['target_path']

notebooks = ['pipeline_01_correction.ipynb',
    'pipeline_02_processing.ipynb',
    'pipeline_03_evaluation.ipynb'
    ]


for notebook in notebooks:
    pm.execute_notebook(input_path=notebook,
        output_path=os.path.join(save_to, notebook),
        parameters=parameters
    )
