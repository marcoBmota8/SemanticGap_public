import argparse
import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold

from src.directories import RESULTS_DIR
from src.utils import load_dataset, record_logs
from src.aux_inf import DIAGNOSES, DISEASE


if __name__ == "__main__":
    
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
    
    record_logs(path=RESULTS_DIR/args.experiment, filename='causal_models_train_output.log')

    # TRAINING
    # Train separate causal models for each target: Diagnosis and Disease (Lung Cancer) itself
    for target_name in DISEASE+DIAGNOSES:
        print(f'Training causal model for {target_name} as target...')
        
        # Load the target
        if args.experiment == 'GT_abstract':
            target = load_dataset()[target_name]
        elif args.experiment == 'GT_sim':
            target = pd.read_parquet(RESULTS_DIR/args.experiment/'X.parquet')[target_name]
        else:
            raise ValueError(rf'Unknown experimental data {args.experiment} ... ¯\_(ツ)_/¯ ')
                
        # Load error terms
        U_train = pd.read_parquet(path=RESULTS_DIR/args.experiment/'U.parquet')
        # Make sure to remove the error term from the disease since in reality we will not have access to it
        U_train.drop(columns=DISEASE, errors='ignore', inplace=True)

        # TUNNING
        # Define hyperparameter search space
        max_features = np.round(np.percentile(np.arange(len(U_train.columns)), [25, 50, 75, 100])).astype(int).tolist() # Get the number of features that represent 25, 50, 75 and 100% of the total dimentsionality
        param_grid = {
            'n_estimators': [10, 100, 500, 1000],
            'max_depth' : [2, 4, 6, 10, 15, 20],
            'max_features': [2] + max_features,
            'min_samples_leaf': [1, 2, 10, 50, 100]
        }
        # Define model architecture
        clf = RandomForestClassifier(
            criterion='gini',
            min_samples_split=2,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            max_samples=None,
            bootstrap=True,
            n_jobs=1,
            random_state=42,
            warm_start=False
        )
        
        # Inner CV for hyperparameter tuning
        print('Tuning model...')
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_search = RandomizedSearchCV(estimator=clf, n_iter=100, param_distributions=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=16, verbose=1)
        cv_search.fit(U_train, target)
        tuning_df = pd.DataFrame(cv_search.cv_results_).set_index('rank_test_score').sort_index()  
            
        # PERFORMANCE
        # Outer CV for unbiased performance
        print('Estimating out-of-sample performance...')
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        nested_score = cross_val_score(cv_search, U_train, target, cv=outer_cv, scoring='roc_auc', verbose=1, n_jobs=1)

        print('Nested CV AUROC: %.3f ± %.3f' % (np.mean(nested_score), np.std(nested_score)))
        print(f'Scores: {nested_score}')
        
        # FINAL MODEL
        # Train the model with best parameters and save it
        clf.set_params(**tuning_df.sort_values(by='mean_test_score', ascending=False)['params'].iloc[0])
        clf.n_jobs = 16
        clf.fit(U_train, target)
        
        with open(RESULTS_DIR/args.experiment/f'model_{target_name.lower()}.pkl', '+wb') as f:
            pickle.dump(clf, f)
        
        