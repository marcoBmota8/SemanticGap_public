import io
import numpy as np
import pandas as pd
import networkx as nx

from contextlib import redirect_stdout
from random import shuffle
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load, parallel_config
from lingam import LiM

from src.utils import recover_logistic_error_terms, load_dataset, save_df
from src.directories import RESULTS_DIR
from src.aux_inf import DIAGNOSES


MMAP_PATH = "/dev/shm/X.mmap" 
_X_SHARED = None


def _get_X_shared():
    global _X_SHARED
    if _X_SHARED is None:
        _X_SHARED = load(MMAP_PATH, mmap_mode='r')
    return _X_SHARED

def run_lim(idx, dis_con, only_global):
    # Draw bootstrap sample
    X = _get_X_shared()
    X_boot = X[idx, :]  # copy local to worker from shared memmap

    # Run LiM model causal disocvery over bootstrapped sample
    model = LiM(
        lambda1=0.1,
        loss_type='logistic' if not np.any(dis_con) else 'mixed',
        max_iter=150,
        h_tol=1e-8,
        rho_max=1e+16,
        w_threshold=0.1,
    )
    
    f = io.StringIO()
    with redirect_stdout(f): # this avoids annoying print from LiM fitting
        model.fit(X_boot, dis_con, only_global=only_global, is_poisson=False)
    B_sample = np.asarray(model.adjacency_matrix_, dtype=float)

    # Check resulting graph acyclicity 
    G = nx.DiGraph(B_sample > 0.0)
    try:
        nx.find_cycle(G)
        # Not a DAG, skip this sample
        return None
    except nx.exception.NetworkXNoCycle:
        # Result is a DAG
        return B_sample

def logistic_errors_from_lim(
    X, dis_con, mcmc_reps=1000, rng=None, only_global=False, n_jobs=-1, verbose=0
):
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    dump(X, MMAP_PATH, compress=0) # Shared memory data structrure

    # Root RNG
    rng = np.random.default_rng() if rng is None else rng

    if mcmc_reps == 1: # Run LiM over the full dataset
        idx = np.arange(X.shape[0])
        shuffle(idx)
        B_hat = run_lim(idx, dis_con, only_global)
    
    else: # We run LiM over bootstrapped repeats
        
        # Spawn independent RNGs for each repetition
        # This is reproducible and avoids correlated streams across processes
        child_rngs = rng.spawn(mcmc_reps)
        idx_list = [child_rngs[i].integers(0, X.shape[0], size=X.shape[0]) for i in range(mcmc_reps)]
        # 1. Run LiM model over bootstrapped samples
        with parallel_config(temp_folder='/dev/shm', max_nbytes='100M', mmap_mode='r'):
            results = Parallel(n_jobs=n_jobs, verbose=verbose, batch_size='auto')(
                delayed(run_lim)(idx_list[i], dis_con, only_global)
                for i in tqdm(range(mcmc_reps))
            )

        # Filter out cyclic graph samples (None) and average just DAGs
        B_hats = np.stack([B_sample for B_sample in results if B_sample is not None], axis=-1)
        B_hat = np.mean(B_hats, axis=-1)
    
    # Filter low weight edges
    B_hat[np.abs(B_hat)<0.1] = 0.0
    
    try:
        return B_hat, B_hats
    except:
        return B_hat


if __name__ == "__main__":
    
    # Recover the DAG with access to all variables except the diagnosis
    X = load_dataset().drop(DIAGNOSES, axis=1)
    f_names = X.columns.tolist()
    n_names = X.index.tolist()
    
    # Binary[0]/continuous[1] indicator array
    dis_con = np.asarray([[0 if X[col].nunique(dropna=True) == 2 else 1 for col in X.columns]])

    # This setting runs LiM several times over bootstrapped sampels of the data
    # B_hat, B_hats = logistic_errors_from_lim(
    #     X=X,
    #     mcmc_reps=100,
    #     dis_con=dis_con,
    #     only_global=False,
    #     n_jobs=10
    #     )
    
    # Run LiM over the full dataset
    B_hat = logistic_errors_from_lim(
        X=X,
        mcmc_reps=1,
        dis_con=dis_con, # Detect and pass continuos and binary variables flags in `X`
        only_global=False,
        n_jobs=1
        )
    
    # Sample logistic error terms from DAG
    U_hat = recover_logistic_error_terms(B_hat, X, dis_con, scale=1.0)
    
    # Convert arrays to dataframes with the appropriate dimension labeling
    B_hat_df = pd.DataFrame(B_hat, columns=f_names, index=f_names)
    U_hat_df = pd.DataFrame(U_hat, columns=f_names, index=n_names)
    
    # Save  results
    try:
        np.save(RESULTS_DIR/'LiM'/'B_samples.npy', B_hats)
    except:
        pass
    save_df(B_hat_df, path=RESULTS_DIR/'LiM', filename='B')
    save_df(U_hat_df, path=RESULTS_DIR/'LiM', filename='U')
    
    
    
    
    
    
    