import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx

from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import binarize
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.directories import DATA_DIR

def load_dataset():
    return pd.read_csv(DATA_DIR/'data.csv', index_col=0)

def logistic_cdf(x, loc=0.0, scale=1.0):
    """
        Logistic CDF  
    """
    z = (x - loc) / scale
    return 1.0 / (1.0 + np.exp(-z))

def logistic_ppf(u, loc=0.0, scale=1.0):
    """ 
        Logistic inverse (quantile)
    """
    return loc + scale * (np.log(u) - np.log1p(-u))

def sample_truncated_logistic(mean, lower, upper, scale=1.0, rng=None, eps=1e-12):
    """
    Draw one sample from Logistic(mean, scale) truncated to (lower, upper).
    Uses inverse-CDF on the truncated interval: U ~ Uniform(F(lower), F(upper)); Z = F^{-1}(U).
    lower/upper can be -np.inf / np.inf.
    """
    rng = np.random.default_rng() if rng is None else rng
    Fl = 0.0 if np.isneginf(lower) else logistic_cdf(lower, loc=mean, scale=scale)
    Fu = 1.0 if np.isposinf(upper) else logistic_cdf(upper, loc=mean, scale=scale)
    # Guard against degeneracy/numerical issues
    Fl = np.clip(Fl, eps, 1.0 - eps)
    Fu = np.clip(Fu, eps, 1.0 - eps)
    if not (Fl < Fu):
        # If the truncation interval is effectively empty, fall back to boundary
        return lower if np.isfinite(lower) else upper
    u = rng.uniform(Fl, Fu)
    return logistic_ppf(u, loc=mean, scale=scale)

def data_augmentation(trunc_vals, scale=1.0, rng=None):
    trunc_vals = np.asarray(trunc_vals)
    assert trunc_vals.ndim == 1, "Passed `trunc_vals` array is not 1-D."
    
    n = trunc_vals.shape[0]
    U = np.empty(n, dtype=float)
    
    for i in range(n):
        if trunc_vals[i] > 0:
            # Truncate to [trunc_val, ∞)
            U[i] = sample_truncated_logistic(mean=trunc_vals[i], lower=trunc_vals[i], upper=np.inf, scale=scale, rng=rng)
        elif trunc_vals[i] < 0:
            # Truncate to (-∞, trunc_val]
            U[i] = sample_truncated_logistic(mean=trunc_vals[i], lower=-np.inf, upper=trunc_vals[i], scale=scale, rng=rng)
        else: # No truncation if trunc_vals=zeros -> Logistic distribution ~(0.0, scale)
            U[i] = sample_truncated_logistic(mean=0.0, lower=-np.inf, upper=np.inf, scale=scale, rng=rng)
    return U
    
def topological_order_from_B(B):
    G = nx.DiGraph()
    p = B.shape[1]
    G.add_nodes_from(range(p))
    js_, is_ = np.where(np.abs(B) > 0)
    G.add_edges_from(zip(js_.tolist(), is_.tolist()))
    return list(nx.topological_sort(G))

def recover_logistic_error_terms(B, X, dis_con, scale, rng=None):
    
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
        
    # Root RNG
    rng = np.random.default_rng() if rng is None else rng

    # Topological order from DAG
    order = topological_order_from_B(B)

    n, p = X.shape
    U_hat = np.full((n, p), np.nan, float)
    
    is_cont = (np.asarray(dis_con).reshape(-1) == 1)
    
    for i in order:
        parents = np.where(np.abs(B[:, i]) > 0)[0]
        
        # Continuous nodes
        if is_cont[i]: 
            if parents.size > 0: # endogenous nodes
                regresor = LinearRegression(fit_intercept=True)
                regresor.fit(X[:, parents], X[:, i])
                eta = regresor.predict(X[:,parents])
                U_hat[:, i] = X[:, i] - eta
            else: # exogenous nodes
                U_hat[:, i] = X[:, i]
                
        # Binary nodes
        else : 
            if parents.size > 0: # endogenous nodes
                regressor = LogisticRegression(penalty=None, fit_intercept=True)
                regressor.fit(X[:, parents], X[:, i])
                eta = regressor.decision_function(X[:, parents]) # log-odds (prediction from parents)
                # Binarization removes information that precludes error term inference
                # We use data-augmentation as solution
                # sample error value from truncated logistic distribution centered at eta (no error model prediction)
                U_hat[:, i] = data_augmentation(
                    trunc_vals=np.where(X[:,i]==1.0, np.maximum(0,-eta), np.minimum(0,-eta)),
                    scale=scale,
                    rng=rng
                    )
            else: # exogenous nodes
                U_hat[:,i] = data_augmentation(
                    trunc_vals=np.zeros(X[:, i].shape),
                    scale=scale,
                    rng=rng
                    )
        
    return U_hat

def sample_data_from_dag(B, n, rng=None):
        
    # Root RNG
    rng = np.random.default_rng() if rng is None else rng

    # 1) Topological order and predictors
    order = topological_order_from_B(B)
    p = B.shape[1]
    
    # 2) Sample error terms independently and compute endogenous variables
    U = np.full((n, p), np.nan, float) # errors
    Z = np.full((n, p), np.nan, float) # latent utilities
    eta = np.full((n, p), np.nan, float) # parents contribution
    X = np.full((n, p), np.nan, float) # endogenous variables observations
    
    for i in order:
        # Sample error terms
        for n_idx in range(n):
            U[n_idx,i] = sample_truncated_logistic(
                mean = 0.0,
                scale = 1.0,
                lower = -np.inf,
                upper = np.inf,
                rng=rng
            )
        # Compute eta, Z and X
        parents = np.where(np.abs(B[:, i]) > 0)[0]
        if parents.size > 0: # Endogenous nodes
            eta[:, i] = X[:, parents] @ B[parents, i]
        else: # Root nodes
            eta[:, i] = np.zeros((n,))
        
        # Utility function
        Z[:, i] = eta[:, i] + U[:, i]
    
        # Observations (binary)
        X[:, i] = binarize(Z[:, i].reshape(1,-1), threshold=0.0, copy=True)
        
    return U, eta, Z, X

def pairwise_mutual_information_with_permutation(
    U,
    discrete=False,
    n_neighbors=5,
    n_permutations=1000,
    alpha=0.05,
    mtc_method='fdr_bh',
    random_state=None,
    clip_zero=True
    ):
    """
    Compute pairwise MI and permutation-test p-values for independence.

    Parameters
    ----------
    U : array-like, shape (n_samples, n_vars)
        Columns are variables to compare.
    discrete : bool or array-like, default False
        If bool, treat all columns as discrete (True) or continuous (False).
        If array-like of length n_vars, a boolean mask per column.
        For mixed types, MI is computed conditioning on the 'target' column type.
    n_neighbors : int, default 5
        k for kNN MI estimators.
    n_permutations : int, default 1000
        Number of label permutations for empirical null.
    alpha : float, default 0.05
        Significance level for declaring dependence after multiple-testing correction.
    mtc_method : {'fdr_bh','bonferroni','sidak','holm','fdr_by', ...}, default 'fdr_bh'
        Method passed to statsmodels.stats.multitest.multipletests.
    random_state : int or None
        RNG seed.
    clip_zero : bool, default True
        Clip tiny negative MI estimates to 0.

    Returns
    -------
    results : dict
        Keys: (i, j) with i < j (and mirrored (j, i)). Values:
        {
          'mi': float,
          'pval': float,
          'pval_adj': float,
          'reject': bool,     # True => evidence of dependence
          'independent': bool,# True => fail to reject independence
          'method': 'classif' or 'regression'
        }
    summary : dict
        {'alpha': alpha, 'mtc_method': mtc_method, 'n_pairs': int,
         'n_reject': int, 'all_independent': bool}
    """
    rng = np.random.default_rng(random_state)
    U = np.asarray(U)
    n, p = U.shape

    # Determine discrete mask
    if isinstance(discrete, (list, tuple, np.ndarray)):
        discrete_mask = np.asarray(discrete, dtype=bool)
        if discrete_mask.shape != (p,):
            raise ValueError("discrete mask must have shape (n_vars,)")
    else:
        discrete_mask = np.full(p, bool(discrete))

    def _mi_xy(x, y, y_is_discrete):
        x = x.reshape(-1, 1)
        if y_is_discrete:
            mi = mutual_info_classif(
                x, y, discrete_features=True, n_neighbors=n_neighbors, random_state=random_state
            )[0]
            return float(mi), 'classif'
        else:
            mi = mutual_info_regression(
                x, y, discrete_features=False, n_neighbors=n_neighbors, random_state=random_state
            )[0]
            return float(mi), 'regression'

    pairs = []
    mi_vals = []
    methods = []
    pvals = []

    # Compute observed MI for each pair and permutation p-values
    for i in range(p):
        for j in range(i + 1, p):
            y_is_discrete = bool(discrete_mask[j])
            mi_obs, method = _mi_xy(U[:, i], U[:, j], y_is_discrete)
            if clip_zero and mi_obs < 0:
                mi_obs = 0.0

            # Permutation null: permute Y (column j)
            y = U[:, j].copy()
            x = U[:, i].copy()
            count = 0
            for k in range(n_permutations):
                y_perm = rng.permutation(y)
                mi_null, _ = _mi_xy(x, y_perm, y_is_discrete)
                if mi_null >= mi_obs - 1e-12:
                    count += 1
                    
            pval = count / n_permutations

            pairs.append((i, j))
            mi_vals.append(mi_obs)
            methods.append(method)
            pvals.append(pval)

    # Multiple-testing correction
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=alpha, method=mtc_method)

    # Build result dict (symmetric)
    results = {}
    n_reject = int(np.sum(reject))
    for (i, j), mi_obs, method, pval, pval_adj, r in zip(pairs, mi_vals, methods, pvals, pvals_adj, reject):
        entry = {
            'mi': mi_obs,
            'pval': float(pval),
            'pval_adj': float(pval_adj),
            'reject': bool(r),            # evidence of dependence
            'independent': bool(not r),   # fail to reject independence
            'method': method
        }
        results[str((i, j))] = entry
        # results[(j, i)] = entry  # mirrored

    summary = {
        'alpha': alpha,
        'mtc_method': mtc_method,
        'n_pairs': len(pairs),
        'n_reject': n_reject,
        'all_independent': (n_reject == 0),
        'n_permutations': n_permutations
    }
    return results, summary

def save_df(df:pd.DataFrame, path:Path, filename:str):
    path.mkdir(exist_ok=True, parents=True)
    filename = filename.split('.')[0]
    filepath = (path/filename).with_suffix('.parquet')
    if filepath.exists():
        raise FileExistsError(f'{filepath}.parquet already exists, manually remove it to overwrite it.')
    else:
        df.to_parquet(filepath, engine='auto')

def save_json(d: dict, path:Path, filename:str):
    path.mkdir(exist_ok=True, parents=True)
    filename = filename.split('.')[0]
    filepath = (path/filename).with_suffix('.json')
    if filepath.exists():
        raise FileExistsError(f'{filepath}.json already exists, manually remove it to overwrite it.')
    else:
        with open(filepath, 'w') as f:
            json.dump(d, f)

def record_logs(path:Path, filename:str='output.log'):
    
    filename, ext = os.path.splitext(filename)
    
    if ext.lower() != '.log':
        filename = filename + '.log'
    
    # Open the log file 
    log_file = open(path/filename, 'w')
    # Redirect stdout and stderr to the file
    sys.stdout = log_file
    sys.stderr = log_file