import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FuncFormatter
from graphviz import Digraph
from shap.plots import beeswarm, violin

from src.directories import VIZ_DIR, RESULTS_DIR
from src.aux_inf import B_gt, f_map, f_names
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

def dag_to_graphviz(B_df:pd.DataFrame, path:Path, filename:str, weight_tol=0.0, fmt="png", engine="dot", rankdir="TB", size="5,5",
                    edge_precision=2, node_style=None, edge_style=None):
    """
    B_df: square pandas DataFrame, index/columns are variable names; value B[j,i] => edge j -> i
    weight_tol: prune edges with |weight| < tol
    fmt: output format (png, pdf, svg, etc.)
    engine: graphviz layout engine ('dot','neato','sfdp','fdp','circo','twopi')
    rankdir: 'LR' left-to-right, 'TB' top-to-bottom
    """
    # Default styles
    node_style = node_style or dict(shape="ellipse", fontsize="14", margin="0.01,0.01", ranksep="1.25")
    edge_style = edge_style or dict(arrowsize="1.0", color="black", penwidth="1.25")
    
    # Prepare graph
    g = Digraph("DAG", format=fmt, engine=engine)
    g.attr(rankdir=rankdir, size=size, splines="true", concentrate="false")
    g.attr("node", **node_style)
    g.attr("edge", **edge_style)

    
    # Add nodes
    for v in B_df.index:
        g.node(str(v))

    # Add directed edges j -> i if B[j,i] passes threshold
    for j in B_df.index:
        for i in B_df.columns:
            w = B_df.loc[j, i]
            if pd.notna(w) and abs(w) >= weight_tol and w != 0:
                label = f"{w:.{edge_precision}f}"
                # Optional color by sign
                color = "seagreen4" if w > 0 else "firebrick3"
                g.edge(str(j), str(i), label=label, color=color)

    outpath = g.render(directory=path.as_posix(), filename=filename, cleanup=True)
    return outpath, g  # outpath is the rendered file path

def no_leading_zero(x, pos):
    # Keep 0 as "0"; for others, format and strip leading zero if between -1 and 1
    if x == 0:
        return "0"
    s = f"{x:.2f}".rstrip("0").rstrip(".")  # trim trailing zeros
    if 0 < abs(x) < 1 and s.startswith("0"):
        s = s[1:]  # remove leading zero
    if -1 < x < 0 and s.startswith("-0"):
        s = "-" + s[2:]  # "-0.25" -> "-.25"
    return s


if __name__ == "__main__":
    
    # CONFIG
    # Define parser
    parser = argparse.ArgumentParser(usage='''
                                     Define what experimental design:
                                        - Ground truth DAG: `GT`
                                        - Linear Mixed Model result DAG: `LiM`
                                        - Other: __folder_name__ (must have set it up beforehand)
                                    and what type of shap plot to generate:
                                        - violin
                                        - beeswarm
                                     ''')
    # Add arguments
    parser.add_argument('-e', '--experiment', type=str, help='Pass the experiment folder name of the DAG to train causal model for.')
    parser.add_argument('-f', '--fig_type', type=str, help='Shap plot to generate.')    
    # Parse arguments
    args = parser.parse_args()
    
    # # DAGs PLOTs
    # # Plot the recovered DAG
    # B_hat_df = pd.read_parquet(RESULTS_DIR/'LiM'/'B.parquet')
    
    # dag_to_graphviz(
    #     B_df=B_hat_df,
    #     path=VIZ_DIR,
    #     filename='recovered_dag',
    #     fmt='pdf'
    # )
    
    # Plot the ground truth graph
    B_gt_df = pd.DataFrame(B_gt, columns=f_names, index=f_names)
    
    dag_to_graphviz(
        B_df=B_gt_df,
        rankdir='TB',
        size='6,6',
        path=VIZ_DIR,
        filename='ground_truth_dag',
        fmt='pdf'
    )
    
    # CAUSAL EFFECTS PLOTs
    # Collect explainer file paths
    expl_files = sorted([e for e in os.listdir(RESULTS_DIR / args.experiment) if e.startswith("explanation_")])

    fig, axes = plt.subplots(1, 3, figsize=(45, 5), constrained_layout=True, sharex=True)

    panel_labels = ["a)", "b)", "c)"]
    
    xmin, xmax = (0,0)
    for i, explainer_n in enumerate(expl_files):

        # Load explainer
        with open(RESULTS_DIR / args.experiment / explainer_n, "rb") as f:
            explanation = pickle.load(f) 
            
        # Compute mav
        mav = np.round(np.mean(abs(explanation.values[:,:,1]), axis=0), decimals=3)

        # shap plot
        if args.fig_type == 'beeswarm':
            # TODO: Verify the beeswarm plots work correctly (code hasnt been tested)
            beeswarm(
                explanation,
                order=explanation.abs.max(0),
                color=plt.get_cmap("RdYlGn"),
                max_display=15,
                show=False,
                ax=axes[i],
            )
        elif args.fig_type == 'violin':
            plt.sca(axes[i]) 
            violin(
                explanation.values[:,:,1],
                features=explanation.data,
                feature_names=[f_map.get(f, '')+f' [{mav[i]}]' for i,f in enumerate(explanation.feature_names)],
                plot_type="layered_violin",
                max_display=15,
                plot_size=(6,4),
                color_bar=(i == len(axes) - 1),
                show=False
                )
            # axes labels
            axes[i].tick_params(axis="y", pad=0, labelsize=9)  
            plt.tick_params(axis="x", which="both", labelsize=7) 
            plt.xlabel("")
        else:
            raise ValueError(f'Unknown args.fig_type==`{args.fig_type}`')
        
        ax_min, ax_max = axes[i].get_xlim()
        if ax_max>xmax:
            xmax=ax_max
        if ax_min<xmin:
            xmin=ax_min

        # Panel label a/b/c in axes coordinates (inside top-left)
        axes[i].text(
            -0.5, 1.02, panel_labels[i],
            transform=axes[i].transAxes,
            fontsize=12, fontweight="bold", va="bottom"
        )  

        # Optional: add a short title based on filename
        axes[i].set_title(explainer_n.replace("explanation_", "").replace(".pkl", "").replace("_", " "), fontsize=12)

    # Set x-axis lims 
    axes[0].set_xlim(xmin, xmax)
    # Set locator
    
    for ax in axes:
        # ax.xaxis.set_major_locator(MultipleLocator(0.01))            # ticks at 0.01 steps
        ax.xaxis.set_major_formatter(FuncFormatter(no_leading_zero))  # custom strings
    # Optional global formatting
    fig.supxlabel("Causal Effect", fontsize=12)
    for ax in axes:
        ax.grid(False)  # cleaner for print

    # Save figure
    fig.savefig(VIZ_DIR / args.experiment/ f"{args.fig_type}_panel.pdf", dpi=250, bbox_inches="tight") 
    plt.close(fig) 
            
        
