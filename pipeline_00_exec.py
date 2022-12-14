import os 
import papermill as pm

preserve_existing = False
path = '/home/mklein/FDA_project'

all_parameters = {
 #  'Lx_Glioblastoma': {
 #       'source_path': '/g/alexandr/smenon/2022-07-13_Glioblastoma/processed_files',
 #       'target_path': os.path.join(path, 'data/Lx_Glioblastoma'),
 #       'condition_name': 'condition',
 #       'well_name': 'rowcol',
 #  },
 # 'Lx_Glioblastoma_filtered': {
 #       'source_path': '/g/alexandr/smenon/2022-07-13_Glioblastoma/processed_files',
 #       'target_path': os.path.join(path, 'data/Lx_Glioblastoma'),
 #       'condition_name': 'condition',
 #       'well_name': 'rowcol',
 #       'keep_conditions': ['TMD_CD95_WT', 'TMD_dM', 'TMD_sM'],
 #       'notebooks': ['pipeline_03_evaluation.ipynb']
 #  },
   'Mx_Seahorse': {
        'source_path': '/home/mklein/Raw Data/220412_Luisa_ScSeahorse_SpaceM',
        'target_path': os.path.join(path, 'data/Mx_Seahorse'),
        'condition_name': 'treatment',
        'well_name': 'rowcol',
    },
    'Lx_Pancreatic_Cancer': {
        'source_path': '/home/mklein/Raw Data/2022-01-31_PancreaticCancer',
        'target_path': os.path.join(path, 'data/Lx_Pancreatic_Cancer'),
        'condition_name': 'condition',
        'well_name': 'rowcol',
    },
}

notebooks = ['pipeline_01_correction.ipynb',
    'pipeline_02_processing.ipynb',
    'pipeline_03_evaluation.ipynb'
    ]

for name, parameters in all_parameters.items():
    save_to = os.path.join(path, 'analysis', name)
    parameters['analysis_path'] = save_to
    
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    
    if not 'notebooks' in parameters.keys():
        parameters['notebooks'] = notebooks      
        
    parameters['project'] = name
    
    for notebook in parameters['notebooks']:
        print('Notebook %s for project %s.'%(notebook, name))
        output_path = os.path.join(save_to, notebook)
        if not preserve_existing or not os.path.exists(output_path):
            try:
                pm.execute_notebook(input_path=os.path.join(path, notebook),
                    output_path=output_path,
                    parameters=parameters
                )
            except Exception as e: 
                print(e)
                print('error completing notebook %s for project %s. Going to the next project.'%(notebook, name))
                break
        else:
            print('Notebook %s for project %s exists and shall not be overwritten in safe mode.'%(notebook, name))

