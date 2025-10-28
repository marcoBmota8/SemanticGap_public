import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

from src.utils import (
    recover_logistic_error_terms,
    save_df,
    save_json,
    load_dataset,
    pairwise_mutual_information_with_permutation
)
from src.aux_inf import B_gt, DIAGNOSES, DISEASE, DISEASE_IDX
from src.directories import RESULTS_DIR, VIZ_DIR

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

if __name__ == "__main__":
    
    X = load_dataset().drop(DIAGNOSES, axis=1)
    f_names = X.columns.tolist() 
    n_names = X.index.tolist()
    
    # Binary[0]/continuous[1] indicator array
    dis_con = np.asarray([[0 if X[col].nunique(dropna=True) == 2 else 1 for col in X.columns]])

    # Recover logistic error terms from DAG
    # B_gt = np.delete(np.delete(B_gt, DISEASE_IDX, axis=0), DISEASE_IDX, axis=1) # Remove disease node
    U = recover_logistic_error_terms(
        B_gt,
        X,
        dis_con,
        scale=1.0
    )
    
    # Convert arrays to dataframes with the appropriate dimension labeling
    U_df = pd.DataFrame(U, columns=f_names, index=n_names)
    
    # Tets for independence (mutual information test)
    results_perm, summary = pairwise_mutual_information_with_permutation(
        U,
        discrete=False,
        n_neighbors=5,
        n_permutations=1000,
        alpha=0.05,
        mtc_method='fdr_bh',
        random_state=None,
        clip_zero=True
    )

    # Save  results
    save_df(U_df, path=RESULTS_DIR/'GT_abstract', filename='U')
    save_json(results_perm, path=RESULTS_DIR/'GT_abstract', filename='ind_test_results.json')
    save_json(summary, path=RESULTS_DIR/'GT_abstract', filename='ind_test_summary.json')
    
    # Plot the error distributions for each feature
    fig, axes = plt.subplots(4,3, figsize=(4,5))
    for i, ax in enumerate(axes.flat):
        try:
            def plot_hist_gaussian_kde(x, label, color, i):
                counts, bins, _ = ax.hist(x, bins=100, density=True, label=label, color=color, alpha=0.6)
            
                mu = np.mean(x)                    
                sigma = np.std(x, ddof=1) 

                # Evaluate Normal PDF at bin centers for alignment
                xc = 0.5 * (bins[1:] + bins[:-1])
                pdf = norm.pdf(xc, loc=mu, scale=sigma) # Gaussian PDF

                ax.plot(xc, pdf, color, lw=1.5, alpha=1, zorder=2)
                ax.set_xlabel(f'{f_names[i].replace('_', ' ')}', fontsize=10)
                ax.tick_params(axis='both', which='both', labelsize=6)
                if i==0:
                    ax.legend(
                        handlelength=0.6, 
                        handleheight=0.5,
                        handletextpad=0.4,
                        borderpad=0.1,
                        labelspacing=0.3,
                        fontsize=9,
                        loc='upper left',
                        frameon=False
                        )
                
            plot_hist_gaussian_kde(U[:,i], r'$U$', 'C0', i)
                        
        except:
            ax.set_visible(False) 
          
    plt.tight_layout(pad=0.5)
    (VIZ_DIR/'GT_abstract').mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ_DIR/'GT_abstract'/'error_term_distributions.pdf', dpi=250)
    
    print('FINISHED')