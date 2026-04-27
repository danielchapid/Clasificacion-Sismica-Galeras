"""
Feature Ranking with Random Forest
----------------------------------
Trains a RandomForest on each partition's context data to calculate and average
feature importances. Generates bar plots for the ranked features.
"""
import os
import sys
sys.dont_write_bytecode = True
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import config_Transformer as config


warnings.filterwarnings("ignore")


# ==============================================================================
# RF CONFIGURATION
# ==============================================================================
# Number of trees in the forest (more trees = more stable ranking)
RF_N_ESTIMATORS  = 500
RF_RANDOM_SEED   = config.RANDOM_SEED
# Number of features to display per plot image
FEATURES_PER_IMG = 18


# ==============================================================================
# UTILITIES
# ==============================================================================

def log(msg, prefix="  >>"):
    print(f"{prefix} {msg}", flush=True)


def sep(title="", char="=", width=58):
    if title:
        pad  = max(0, width - len(title) - 2)
        left = pad // 2
        print(f"\n  {char*left} {title} {char*(pad-left)}", flush=True)
    else:
        print(f"  {char*width}", flush=True)


# ==============================================================================
# LOAD CONTEXT DATA BY PARTITION
# ==============================================================================

def load_context(partition: int):
    """Reads the context of a partition and returns X, y, feat_cols."""
    # Use existing directories (Partition_X and context) instead of config.py defaults
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ctx_path = os.path.join(
        base_dir, "data_processed", "partitions", 
        f"Partition_{partition}", "context_80", "features_context.parquet"
    )

    if not os.path.exists(ctx_path):
        log(f"[ERROR] Not found: {ctx_path}", prefix="  !!")
        return None, None, None

    df   = pd.read_parquet(ctx_path)
    META = {config.LABEL_COL, "event_id", "event_group", "station_code", "file_name"}
    feat_cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]

    y = df[config.LABEL_COL].astype(str).values

    X_raw = df[feat_cols].values.astype(np.float32)
    X_raw[np.isinf(X_raw)] = np.nan
    nan_means = np.nanmean(X_raw, axis=0)
    nan_means[np.isnan(nan_means)] = 0.0
    mask = np.isnan(X_raw)
    X_raw[mask] = np.take(nan_means, np.where(mask)[1])

    return X_raw, y, feat_cols


# ==============================================================================
# TRAIN RF AND GET IMPORTANCES
# ==============================================================================

def partition_importances(X, y, partition):
    """Trains RF on a partition and returns importance array."""
    sep(f"PARTITION {partition} - Random Forest", char="-")

    dist = dict(sorted(Counter(y).items()))
    log(f"Rows     : {len(X):,}")
    log(f"Features : {X.shape[1]}")
    for cls, cnt in dist.items():
        log(f"  {cls}: {cnt:,}")

    rf = RandomForestClassifier(
        n_estimators = RF_N_ESTIMATORS,
        random_state = RF_RANDOM_SEED,
        n_jobs       = -1,
        class_weight = "balanced",
    )
    rf.fit(X, y)

    acc = rf.score(X, y)
    log(f"RF Accuracy (train): {acc*100:.1f}%")

    return rf.feature_importances_


# ==============================================================================
# SPLIT PLOTS INTO MULTIPLE IMAGES
# ==============================================================================

def generate_plots(ranking_df, out_dir):
    """
    Generates one plot exclusively for important features (green, >= global mean),
    and separates the remaining less important features (red) into subsequent images.
    """
    mean_global = ranking_df["mean_importance"].mean()
    
    df_important = ranking_df[ranking_df["mean_importance"] >= mean_global].copy()
    df_low       = ranking_df[ranking_df["mean_importance"] < mean_global].copy()

    paths = []

    def plot_chunk(chunk, title_suffix, filename, fill_color):
        if chunk.empty: return
        fig, ax = plt.subplots(figsize=(12, max(6, len(chunk) * 0.45)))

        bars = ax.barh(
            range(len(chunk)),
            chunk["mean_importance"].values,
            xerr=chunk["std_importance"].values,
            color=fill_color,
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
        )

        ax.set_yticks(range(len(chunk)))
        ax.set_yticklabels(chunk["feature"].values, fontsize=10)
        ax.invert_yaxis()

        ax.set_xlabel("Importance (Gini) — 4 Partitions Average", fontsize=11)
        ax.set_title(f"Feature Importance — {title_suffix}", fontsize=13, fontweight="bold", pad=12)

        ax.axvline(x=mean_global, color="#3498db", linestyle="--",
                   linewidth=1.5, label=f"Global mean = {mean_global:.4f}")
        ax.legend(fontsize=10, loc="lower right")

        for i, (val, std) in enumerate(zip(chunk["mean_importance"].values, chunk["std_importance"].values)):
            ax.text(val + std + 0.0005, i, f"{val:.4f}", va="center", fontsize=9, color="#333")

        plt.tight_layout(pad=1.5)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        log(f"Plot saved: {path}")

    # Plot 1: Important Features (Green)
    title_top = f"Top {len(df_important)} Features (IMPORTANT)"
    plot_chunk(df_important, title_top, "feature_importance_1_TOP.png", "#2ecc71")

    # Plots 2+: Lower Importance Features (Red)
    n_rest = len(df_low)
    n_imgs = int(np.ceil(n_rest / FEATURES_PER_IMG))

    for img_idx in range(n_imgs):
        start = img_idx * FEATURES_PER_IMG
        end   = min(start + FEATURES_PER_IMG, n_rest)
        chunk = df_low.iloc[start:end]
        
        # Access original ranking index for title
        rank_start = chunk.index[0]
        rank_end   = chunk.index[-1]
        
        title_rest = f"Rank {rank_start} to {rank_end} (Low Importance)"
        filename   = f"feature_importance_{img_idx+2}_REST.png"
        plot_chunk(chunk, title_rest, filename, "#e74c3c")

    return paths


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    out_dir = os.path.join(config.RESULTS_DIR, "feature_importance")
    os.makedirs(out_dir, exist_ok=True)

    W = 58
    print(f"\n  {'='*W}")
    print(f"  FEATURE IMPORTANCE - Random Forest × {config.N_PARTITIONS} Partitions")
    print(f"  Trees: {RF_N_ESTIMATORS}  |  Seed: {RF_RANDOM_SEED}")
    print(f"  {'='*W}\n")

    # -- Train RF on each partition ----------------------------------------------
    all_importances = []
    feat_cols       = None

    for partition in range(1, config.N_PARTITIONS + 1):
        X, y, cols = load_context(partition)
        if X is None:
            log(f"[SKIP] Partition {partition}: no data.", prefix="  !!")
            continue

        if feat_cols is None:
            feat_cols = cols
        else:
            assert cols == feat_cols, \
                f"Partition {partition} has different columns than partition 1"

        imp = partition_importances(X, y, partition)
        all_importances.append(imp)

    if not all_importances:
        print("  [ERROR] Could not train any partition.")
        sys.exit(1)

    # -- Average importances ------------------------------------------------
    sep("AVERAGED RANKING", char="=")

    imp_matrix = np.array(all_importances)   # shape: (n_partitions, n_features)
    imp_mean   = imp_matrix.mean(axis=0)
    imp_std    = imp_matrix.std(axis=0)

    ranking_df = pd.DataFrame({
        "feature"           : feat_cols,
        "mean_importance"   : imp_mean,
        "std_importance"    : imp_std,
    })

    # Add individual partition importance
    for i, partition in enumerate(range(1, len(all_importances) + 1)):
        ranking_df[f"Partition_{partition}"] = imp_matrix[i]

    ranking_df = ranking_df.sort_values("mean_importance", ascending=False)\
                           .reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1   # ranking starts at 1
    ranking_df.index.name = "rank"

    # -- Display on console ----------------------------------------------------
    mean_global = ranking_df["mean_importance"].mean()

    print(f"\n  {'Rank':<6}  {'Feature':<35}  {'Mean':>10}  {'Std':>8}  {'Status'}")
    print(f"  {'-'*6}  {'-'*35}  {'-'*10}  {'-'*8}  {'-'*12}")

    above_mean = 0
    for rank, row in ranking_df.iterrows():
        status = "[v] IMPORTANT" if row["mean_importance"] >= mean_global else "[x] low"
        if row["mean_importance"] >= mean_global:
            above_mean += 1
        print(f"  {rank:<6}  {row['feature']:<35}  "
              f"{row['mean_importance']:>10.5f}  "
              f"{row['std_importance']:>8.5f}  {status}")

    print(f"\n  {'-'*75}")
    print(f"  Total features       : {len(ranking_df)}")
    print(f"  Global mean          : {mean_global:.5f}")
    print(f"  Features >= mean     : {above_mean}")
    print(f"  Features < mean      : {len(ranking_df) - above_mean}")

    # -- Generate split plots --------------------------------------------
    sep("PLOTS", char="-")
    generate_plots(ranking_df, out_dir)

    # -- Final summary ---------------------------------------------------------
    print(f"\n  {'='*W}")
    print(f"  FEATURE IMPORTANCE COMPLETED")
    print(f"  {'='*W}")
    print(f"\n  Results in: {out_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Review the generated plots")
    print(f"    2. Decide which features to keep in config_extract.py")
    print(f"    3. Re-run extract_features.py with only the important features")
    print(f"    4. Run Transformer.py with the newly extracted data")
    print(f"  {'='*W}\n")


if __name__ == "__main__":
    main()

