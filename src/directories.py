from pathlib import Path

current_file = Path(__file__).resolve()

SRC_DIR = current_file.parent 
ROOT_DIR =  SRC_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
VIZ_DIR = ROOT_DIR / 'viz'
RESULTS_DIR = ROOT_DIR / 'results'
REPRODUCE_DIR = ROOT_DIR / 'reproduce'
FIG_DIR = ROOT_DIR / 'figures'