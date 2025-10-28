from src.directories import DATA_DIR, VIZ_DIR, RESULTS_DIR, REPRODUCE_DIR, FIG_DIR

# Create the appropriate folder structure if not present
DATA_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
