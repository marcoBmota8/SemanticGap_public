import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

from src.directories import RESULTS_DIR, VIZ_DIR
from src.aux_inf import f_available
from src.utils import load_dataset, record_logs

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

def compute_noise_prob(sample_idx, beta=0.5):
    '''
        Compute the probability of fliping a label (binary noise) based on 
            -u_e : Uncertainty from low representation (leaf frequency)
            -u_h : Hard cases uncertainty
        ------------------------------------
        Args:
        - Beta regulates the balance of how much we weight
          either type of uncertainty. 
          Larger beta gives more relevance to uncertainty from
          leaf frequency versus hard cases.
          Beta should be large in heterogenous diseases with many 
          and possibly rare phenotypes, while it should be low in 
          those unspecifc diseases with generic symptomatology easily confused
          with other conditions and/or healthy status.
    '''
    leaf = leaf_indices[sample_idx]
    u_e = leaf_uncertainty[leaf] 
    p = probs[sample_idx]
    u_h = 1 - abs(p - 0.5) * 2  # Borderline uncertainty (max at 0.5)
    noise_prob = beta * u_e + (1 - beta) * u_h
    return np.clip(noise_prob, 0, 1) # probability limits

# CONFIG
# Define parser 
parser = argparse.ArgumentParser(usage='What type of data to use to build the labeller: `GT_abstract` or `GT_sim` (for simulated).')
# Add arguments
parser.add_argument('-d', '--data', type=str, help='What type of data to use to build the labeller: `GT_abstract` or `GT_sim` (for simulated).')
# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":
    
    record_logs(RESULTS_DIR/args.data, f'labeler_output.log')
    
    # Load data and get the clinically available featues only
    if args.data == "GT_abstract":
        X = load_dataset()
    elif args.data == "GT_sim":
        X = pd.read_parquet(RESULTS_DIR/args.data/'X.parquet')
    else:
        raise ValueError(rf'Unknown experimental data {args.data} ... ¯\_(ツ)_/¯ ')
    
    y_true = X['Lung_Cancer'] # True disease 
    X_dx=X.loc[:,f_available] # Data to use for diagnosing
    
    # Train labeller decission tree
    dt = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=3, # Medium complexity
        max_features=None,
        min_samples_leaf=2,
        random_state=42
    )
    
    dt.fit(X_dx,y_true)
    
    # Plot the learnt labeller
    fig = plt.figure(figsize=(7,5), dpi=300)
    plot_tree(
        decision_tree=dt,
        max_depth=None,
        feature_names=X_dx.columns.tolist(),
        class_names=['Negative', 'Cancer Dx'],
        fontsize=9,
        proportion=True,
        precision=2,
        label='none',
        filled=True
              )
    fig.savefig(VIZ_DIR/args.data/f'labeller.pdf')
    
    # Get clean diagnosis labels
    y_pred = dt.predict(X_dx)
    # Accuracy
    acc_clean = accuracy_score(y_true, y_pred)
    print(f'Clean labels disease-surrogate accuracy: {np.mean(acc_clean)}.')
    
    # Add noise to labels
    # Get leaf node indices for each sample
    leaf_indices = dt.apply(X_dx)

    # Count samples per leaf node
    unique_leaves, counts = np.unique(leaf_indices, return_counts=True)
    leaf_counts = dict(zip(unique_leaves, counts))

    # Get predicted class probabilities per sample
    probs = dt.predict_proba(X_dx)[:, 1]  # Probability of positive class

    # Define uncertainty based on leaf sample size
    alpha = 1  # smoothing constant
    leaf_uncertainty = {leaf: 1 / (count + alpha) for leaf, count in leaf_counts.items()}

    # Generate noisy labels by flipping with noise_prob
    rng = np.random.default_rng(42)
    y_noisy = y_pred.copy()
    for i in range(len(y_pred)):
        noise_p = compute_noise_prob(i, beta=0.7)
        if rng.random() < noise_p:
            y_noisy[i] = 1 - y_pred[i]  # Flip 
    
    # Accuracy
    acc_noisy = accuracy_score(y_true, y_noisy)
    print(f'Noisy labels disease-surrogate accuracy: {acc_noisy}.')
    
    # Save the labeller model
    with open(RESULTS_DIR/f'labeller.pkl', '+wb') as f:
        pickle.dump(dt, f)
        
    # Save the extended dataset
    X['Diagnosis_clean'] = y_pred
    X['Diagnosis_noisy'] = y_noisy
    
    if args.data == "GT_abstract": 
        X.to_csv('./data/data.csv')
    elif args.data == "GT_sim":
        X.to_parquet(RESULTS_DIR/args.data/'X.parquet')
    