import os
import argparse
import pickle

import pandas as pd

from shap import TreeExplainer

from src.directories import RESULTS_DIR
from src.utils import record_logs
from src.aux_inf import DISEASE

# CONFIG
# Define parser 
parser = argparse.ArgumentParser(usage='''
                                    Define what experimental design to train a causal model for: 
                                    - Ground truth DAG: `GT`
                                    - Linear Mixed Model result DAG: `LiM`
                                    - Other: __folder_name__ (must have set it up beforehand)
                                    ''')
# Add arguments
parser.add_argument('-e', '--experiment', type=str, help='Pass the experiment folder name of the DAG to train causal model for.')
# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":
    
    record_logs(path=RESULTS_DIR/args.experiment, filename='causal_effect_estimation_output.log')
    
    # COMPUTE SHAPLEY VALUES OVER THE TRAINING DATA 
    
    # Load error terms
    U_train = pd.read_parquet(path=RESULTS_DIR/args.experiment/'U.parquet') 
    # Make sure to remove the error term from the disease since in reality we will not have access to it
    U_train.drop(columns=DISEASE, errors='ignore', inplace=True)
    
    # Compute Shapey values for all the models in the experiment directory
    for model_n in [model_n for model_n in os.listdir(RESULTS_DIR/args.experiment) if model_n.startswith("model_")]:
        print(f'Compute Shapley valyes for {model_n}...')
        # Load Causal model
        with open(RESULTS_DIR/args.experiment/model_n, 'rb') as f:
            model = pickle.load(f)
            
        # Estimate Shapley values 
        explainer = TreeExplainer(
            model=model,
            data=None,
            model_output='raw', # For sklearn classifiers this is probability
            feature_perturbation='tree_path_dependent'
        )
        explanation = explainer(U_train)
        shaps = pd.DataFrame(
            data = explanation.values[:,:,1],
            index = U_train.index,
            columns = U_train.columns
        )
        
        # Save Shapley values and explaination
        filename = model_n.split('.')[0].split('model_')[1]
        shaps.to_parquet(RESULTS_DIR/args.experiment/f'shaps_{filename}.parquet')
        with open(RESULTS_DIR/args.experiment/f'explanation_{filename}.pkl', 'wb') as f:
            pickle.dump(explanation, f)