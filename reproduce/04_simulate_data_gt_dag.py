import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

from src.utils import sample_data_from_dag, save_df, save_json, pairwise_mutual_information_with_permutation
from src.aux_inf import B_gt, f_names
from src.directories import RESULTS_DIR, VIZ_DIR

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

if __name__ == "__main__":
    
    # Sample logistic error terms from DAG
    U, eta, Z, X = sample_data_from_dag(
        B_gt,
        n=5000
        )

    # Compute mutual information
    results, summary = pairwise_mutual_information_with_permutation(
        U,
        discrete=False,
        n_neighbors=5,
        n_permutations=1000,
        alpha=0.05,
        mtc_method='fdr_bh',
        random_state=None,
        clip_zero=True
    )

    # Convert arrays to dataframes with the appropriate dimension labeling
    U_df = pd.DataFrame(U, columns=f_names)
    # eta = pd.DataFrame(eta, columns=f_names)
    Z_df = pd.DataFrame(Z, columns=f_names)
    X_df = pd.DataFrame(X, columns=f_names)

    # Save  results
    (RESULTS_DIR/'GT_sim').mkdir(parents=True, exist_ok=True)
    save_df(U_df, path=RESULTS_DIR/'GT_sim', filename='U')
    # save_df(eta, path=RESULTS_DIR/'GT_sim', filename='eta')
    save_df(Z_df, path=RESULTS_DIR/'GT_sim', filename='Z')
    save_df(X_df, path=RESULTS_DIR/'GT_sim', filename='X')
    
    save_json(results, path=RESULTS_DIR/'GT_sim', filename='ind_test_results.json')
    save_json(summary, path=RESULTS_DIR/'GT_sim', filename='ind_test_summary.json')
    
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
                
            plot_hist_gaussian_kde(U[:,i], r'$U_i$', 'C0', i)
            
        except:
            ax.set_visible(False) 
          
    plt.tight_layout(pad=0.5)
    fig.savefig(VIZ_DIR/'sim_error_term_distributions.pdf', dpi=250)
    
    print('FINISHED')