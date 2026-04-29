"""
TabPFN Transformer Classification System - Configuration
"""
import os

# --- Base Paths ---
# Project root is one level above this script's folder
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Input directory for processed partitions
FOLDS_DIR    = os.path.join(BASE_DIR, "data_processed", "partitions")
# Output directory for evaluation results
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

# --- Partition Path Builder ---
def get_partition_paths(partition: int) -> dict:
    """Returns all relevant paths for a specific partition."""
    partition_data = os.path.join(FOLDS_DIR,   f"Partition_{partition}")
    partition_res  = os.path.join(RESULTS_DIR, f"partition_{partition}")
    return {
        "context_parquet" : os.path.join(partition_data, "context_80", "features_context.parquet"),
        "test_dir"        : os.path.join(partition_data, "test_20"),
        "results_dir"     : partition_res,
        "embeddings_dir"  : os.path.join(partition_res, "embeddings"),
        "confusion_matrix": os.path.join(partition_res, "confusion_matrix.png"),
        "metricas_csv"    : os.path.join(partition_res, "metricas.csv"),
    }

# --- Partitions ---
# Must match N_PARTITIONS in config_extract.py
N_PARTITIONS = 4

# --- Columns ---
LABEL_COL = "label"

# --- TabPFN Configuration (tabpfn >= 6.x) ---
# Internal transformer ensembles
TABPFN_ESTIMATORS    = 4
# Minimum guaranteed samples per class in context
TABPFN_MIN_PER_CLASS = 100
# Native TabPFN row limit
TABPFN_MAX_ROWS      = 10000
# Batch size for predict_proba (prevents OOM)
EMBED_BATCH_SIZE     = 500

# --- Class Balancing (applied on probability vectors before argmax) ---
# Formula: w_c = (N / (C * n_c)) ^ alpha
# alpha=0.0 -> no balancing
# alpha=0.5 -> moderate balancing
# alpha=1.0 -> full balancing
CLASS_WEIGHT_ALPHA = 0.0

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Visualization ---
FIG_DPI        = 150
CONFUSION_CMAP = "Blues"