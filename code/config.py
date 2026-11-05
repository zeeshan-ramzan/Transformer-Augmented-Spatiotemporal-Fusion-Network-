"""
Central configuration: folder layout, dataset columns and evaluation protocol.

Folder layout (relative to this file):

    TASTFN_Replication/
        code/        <- all Python files (this folder)
        dataset/     <- place nthri_tea_yield.csv here
        results/     <- created automatically; one sub-folder per algorithm
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"
RESULTS_DIR = ROOT_DIR / "results"
DATA_FILE = DATASET_DIR / "nthri_tea_yield.csv"

# ----------------------------------------------------------------------------
# Dataset columns (see dataset/README.txt)
# ----------------------------------------------------------------------------
PLOT_COL = "plot_id"          # plot identifier (alias accepted: farm_id)
DATE_COL = "date"             # YYYY-MM (or any date parsable by pandas)
TARGET_COL = "yield"          # fresh-leaf yield, kg/ha

AGRO_FEATURES = ["min_temp", "max_temp", "mean_temp",
                 "humidity", "rainfall", "solar_radiation"]
NDVI_FEATURES = ["ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max"]
FEATURES = AGRO_FEATURES + NDVI_FEATURES
TEMPERATURE_GROUP = ["min_temp", "max_temp", "mean_temp"]

# Optional columns used only by the NDVI-source sensitivity analysis
L78_NDVI_FEATURES = ["l78_ndvi_mean", "l78_ndvi_std", "l78_ndvi_min", "l78_ndvi_max"]
S2_NDVI_FEATURES = ["s2_ndvi_mean", "s2_ndvi_std", "s2_ndvi_min", "s2_ndvi_max"]
GAPFILL_COL = "ndvi_gapfilled"   # 0/1 flag

FEATURE_LABELS = {
    "min_temp": "Min temperature", "max_temp": "Max temperature",
    "mean_temp": "Mean temperature", "humidity": "Relative humidity",
    "rainfall": "Rainfall", "solar_radiation": "Solar radiation",
    "ndvi_mean": "NDVI mean", "ndvi_std": "NDVI standard deviation",
    "ndvi_min": "NDVI minimum", "ndvi_max": "NDVI maximum",
}

# ----------------------------------------------------------------------------
# Evaluation protocol (as in the paper)
# ----------------------------------------------------------------------------
# 1) Independent test year: the last year in the data (2025) is held out and
#    used only once, to evaluate models trained on all earlier years.
# 2) Leave-one-year-out cross-validation (LOYO-CV) over the remaining years
#    (2015-2024). Inside each outer fold the most recent training year is the
#    inner validation year (early stopping, feature/hyperparameter selection).
# 3) Stratified 10-fold CV (random split, stratified on yield deciles) is also
#    reported for continuity, as an optimistic upper bound.
TEST_YEAR = None              # None -> use the last year present in the data
N_STRATIFIED_FOLDS = 10
RANDOM_STATE = 42

# Seasons (plucking months March-November)
SEASONS = {"Spring flush (Mar-May)": [3, 4, 5],
           "Summer season (Jun-Sep)": [6, 7, 8, 9],
           "Autumn flush (Oct-Nov)": [10, 11]}

# ----------------------------------------------------------------------------
# TASTFN hyperparameters (Table 4 of the paper)
# ----------------------------------------------------------------------------
TASTFN_PARAMS = dict(
    d_model=64, n_heads=8, d_ff=256, n_layers=2, context=6,
    decoder_widths=(128, 64, 32), dropout=0.30, attn_dropout=0.10,
    lr=1e-3, weight_decay=1e-4, t0=50, eta_min=1e-5,
    batch_size=32, max_epochs=500, patience=30,
)
