"""
TabPFN v2 Transformer - Context Training + Partition Evaluation
===============================================================
For each partition:
  1. Reads features_context.parquet
  2. Imputes NaN and computes context means
  3. Trains TabPFN v2 with the context (fit)
  4. Evaluates test parquets with argmax + optional balancing
  5. Saves probability vector TXT, confusion matrix, and metrics CSV

Classification: argmax(probability_vector * balance_weights)
Balance       : w_c = (N / (C * n_c)) ^ alpha  (alpha=0 -> no balance)

Usage:
    python evaluar.py --partition 1
    python evaluar.py --partition 2
    python evaluar.py --partition 3
    python evaluar.py --partition 4
"""

import os
import sys
import argparse
sys.dont_write_bytecode = True
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabpfn import TabPFNClassifier

import config_Transformer as config

warnings.filterwarnings("ignore")


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
# 1. CONTEXT - Read, impute NaN, train TabPFN
# ==============================================================================

def preparar_contexto(fold: int) -> tuple:
    """Reads context, imputes NaN, trains TabPFN.
    Returns: (model, nan_means, feat_cols, all_classes, y_ctx)
    """
    paths    = config.get_partition_paths(fold)
    ctx_path = paths["context_parquet"]

    if not os.path.exists(ctx_path):
        log(f"[ERROR] Not found: {ctx_path}", prefix="  !!")
        log("Run extract_features.py first.", prefix="  !!")
        sys.exit(1)

    sep(f"PARTITION {fold} - CONTEXT TabPFN", char="-")

    df    = pd.read_parquet(ctx_path)
    META  = {config.LABEL_COL, "event_id", "station_code", "file_name"}
    feat_cols = [
        c for c in df.columns
        if c not in META and pd.api.types.is_numeric_dtype(df[c])
    ]

    log(f"File     : {ctx_path}")
    log(f"Rows    : {df.shape[0]:,}")
    log(f"Features : {len(feat_cols)}")

    y = df[config.LABEL_COL].astype(str).values

    # Impute NaN with context means (prevents data leakage in test)
    X_raw = df[feat_cols].to_numpy(dtype=np.float32)
    X_raw[np.isinf(X_raw)] = np.nan
    nan_means = np.nanmean(X_raw, axis=0)
    nan_means[np.isnan(nan_means)] = 0.0
    mask = np.isnan(X_raw)
    X_raw[mask] = np.take(nan_means, np.where(mask)[1])
    X = X_raw

    # Context distribution
    all_classes = sorted(np.unique(y).tolist())
    dist        = dict(sorted(Counter(y).items()))

    print(f"\n  {'Class':<10}  {'Rows':>8}  {'%':>7}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*7}")
    for cls, cnt in dist.items():
        pct = cnt / len(y) * 100
        print(f"  {cls:<10}  {cnt:>8,}  {pct:>6.1f}%", flush=True)
    print(f"  {'-'*10}  {'-'*8}")
    print(f"  {'TOTAL':<10}  {len(y):>8,}")

    # Stratified sampling if exceeding TabPFN limit
    if len(X) > config.TABPFN_MAX_ROWS:
        X, y = _stratified_sample(X, y)
        print(f"\n  Using {len(X):,} rows in TabPFN context "
              f"(limit={config.TABPFN_MAX_ROWS:,})", flush=True)
    else:
        log(f"Using all {len(X):,} context rows "
            f"(< limit {config.TABPFN_MAX_ROWS:,})")

    # Train TabPFN v2
    model = TabPFNClassifier(
        n_estimators             = config.TABPFN_ESTIMATORS,
        random_state             = config.RANDOM_SEED,
        n_preprocessing_jobs     = -1,
        ignore_pretraining_limits= True,
    )
    model.fit(X, y)
    log(f"Probability vector order : {list(model.classes_)}")
    log(f"Context ready.")

    return model, nan_means, feat_cols, all_classes, y


def _stratified_sample(X, y):
    classes, counts = np.unique(y, return_counts=True)
    n_classes   = len(classes)
    guaranteed  = config.TABPFN_MIN_PER_CLASS * n_classes
    remaining   = max(0, config.TABPFN_MAX_ROWS - guaranteed)
    rng         = np.random.default_rng(config.RANDOM_SEED)
    selected    = []

    for cls, cnt in zip(classes, counts):
        cls_idx      = np.where(y == cls)[0]
        n_guaranteed = min(config.TABPFN_MIN_PER_CLASS, cnt)
        n_extra      = int(remaining * (cnt / len(y)))
        n_total      = min(n_guaranteed + n_extra, cnt)
        chosen       = rng.choice(cls_idx, size=n_total, replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(selected)
    if len(selected) > config.TABPFN_MAX_ROWS:
        selected = rng.choice(selected, size=config.TABPFN_MAX_ROWS, replace=False)

    return X[selected], y[selected]


# ==============================================================================
# 2. CLASS BALANCING
# ==============================================================================

def compute_class_weights(y_context: np.ndarray, all_classes: list) -> np.ndarray:
    if config.CLASS_WEIGHT_ALPHA == 0.0:
        log(f"Balancing disabled (alpha=0.0)")
        return np.ones(len(all_classes), dtype=np.float32)

    N      = len(y_context)
    C      = len(all_classes)
    counts = Counter(y_context)

    print(f"\n  Balance (alpha={config.CLASS_WEIGHT_ALPHA}):")
    print(f"  {'Class':<10}  {'n_c':>8}  {'w_c':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}")

    weights = []
    for cls in all_classes:
        n_c   = counts.get(cls, 1)
        w     = float((N / (C * n_c)) ** config.CLASS_WEIGHT_ALPHA)
        weights.append(w)
        print(f"  {cls:<10}  {n_c:>8,}  {w:>8.4f}")

    return np.array(weights, dtype=np.float32)


# ==============================================================================
# 3. METRICS
# ==============================================================================

def compute_metrics(cm_dict, all_classes):
    classes = list(all_classes)
    n       = len(classes)
    idx     = {c: i for i, c in enumerate(classes)}
    cm      = np.zeros((n, n), dtype=np.int64)
    support = {c: 0 for c in classes}

    for true_cls, pred_counts in cm_dict.items():
        if true_cls not in idx:
            continue
        for pred_cls, count in pred_counts.items():
            support[true_cls] += count
            if pred_cls in idx:
                cm[idx[true_cls], idx[pred_cls]] += count

    total    = sum(support.values())
    correct  = sum(cm[i, i] for i in range(n))
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    for i, cls in enumerate(classes):
        TP  = int(cm[i, i])
        FP  = int(cm[:, i].sum()) - TP
        FN  = support[cls] - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        denom_f1  = 2 * TP + FP + FN
        f1        = (2 * TP) / denom_f1 if denom_f1 > 0 else 0.0
        per_class[cls] = {
            "TP": TP, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall,
            "f1": f1, "support": int(support[cls]),
        }

    f1_macro    = float(np.mean([per_class[c]["f1"] for c in classes])) if classes else 0.0
    total_sup   = sum(per_class[c]["support"] for c in classes)
    f1_weighted = (
        sum(per_class[c]["f1"] * per_class[c]["support"] for c in classes) / total_sup
        if total_sup > 0 else 0.0
    )
    return cm, classes, accuracy, per_class, f1_macro, f1_weighted


# ==============================================================================
# 4. SAVE PROBABILITY VECTORS TO TXT
# ==============================================================================

def save_embeddings_txt(true_class, embeddings, all_classes, out_path):
    N          = len(embeddings)
    header_emb = ", ".join(f"P({c})" for c in all_classes)
    argmax_idx = np.argmax(embeddings, axis=1)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"  True label              : {true_class}\n")
        f.write(f"  Probability vector      : [{header_emb}]\n")
        f.write(f"  Total vectors           : {N:,}\n")
        f.write("=" * 72 + "\n\n")
        for i in range(N):
            vec      = embeddings[i]
            pred_cls = all_classes[int(argmax_idx[i])]
            symbol   = "[v]" if pred_cls == true_class else "[x]"
            vec_str  = "[" + ", ".join(f"{v:.3f}" for v in vec) + "]"
            f.write(f"  {i+1:>7}/{N:<7} {vec_str}  -> argmax: {pred_cls} {symbol}\n")
        f.write("\n" + "=" * 72 + "\n")


# ==============================================================================
# 5. CONFUSION MATRIX
# ==============================================================================

def save_confusion_matrix(cm, classes, out_path, fold):
    n_cls    = len(classes)
    fig_side = max(6.0, n_cls * 2.2)
    font_sz  = max(11, min(18, int(48 / max(n_cls, 1))))
    tick_sz  = max(9,  min(14, font_sz - 2))

    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    im      = ax.imshow(cm.astype(float), interpolation="nearest",
                        cmap=config.CONFUSION_CMAP)
    fig.colorbar(im, ax=ax, fraction=0.040, pad=0.04).ax.tick_params(
        labelsize=tick_sz)
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=tick_sz)
    ax.set_yticklabels(classes, fontsize=tick_sz)

    thresh = cm.max() / 2.0
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    fontsize=font_sz, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black", zorder=3)

    ax.set_xlabel("Prediction",  fontsize=tick_sz, labelpad=10)
    ax.set_ylabel("True Label",  fontsize=tick_sz, labelpad=10)
    ax.set_title(f"Confusion Matrix - Partition {fold}  |  "
                 f"alpha={config.CLASS_WEIGHT_ALPHA}",
                 fontsize=tick_sz, pad=14)
    plt.tight_layout(pad=2.5)
    fig.savefig(out_path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 6. EVALUATION LOOP
# ==============================================================================

def evaluar_fold(fold: int) -> dict:
    paths = config.get_partition_paths(fold)
    os.makedirs(paths["results_dir"],    exist_ok=True)
    os.makedirs(paths["embeddings_dir"], exist_ok=True)
    
    # Anti-race condition for cloud drives (OneDrive / Google Drive)
    time.sleep(0.1)

    # -- Prepare context and train TabPFN -------------------------------------
    model, nan_means, feat_cols, all_classes, y_ctx = preparar_contexto(fold)

    # -- Balance weights -------------------------------------------------------
    class_weights = compute_class_weights(y_ctx, all_classes)

    # -- Test files -------------------------------------------------------------
    eval_files = sorted(Path(paths["test_dir"]).glob("*.parquet"))
    if not eval_files:
        log(f"[ERROR] No parquets in: {paths['test_dir']}", prefix="  !!")
        sys.exit(1)

    log(f"Test files : {[f.name for f in eval_files]}")

    NON_FEAT = {"label", "class", config.LABEL_COL.lower(),
                "event_id", "station_code", "file_name"}
    cm_dict  = {}

    sep(f"PARTITION {fold} - EVALUATION", char="-")

    for eval_path in eval_files:
        true_class = eval_path.stem
        print(f"\n  {'-'*58}", flush=True)
        print(f"  {eval_path.name}", flush=True)

        df_raw   = pd.read_parquet(str(eval_path))
        df_blind = df_raw.drop(
            columns=[c for c in df_raw.columns if c.lower() in NON_FEAT],
            errors="ignore"
        )

        log(f"True label : {true_class}")
        log(f"Rows       : {df_blind.shape[0]:,}")

        missing = [c for c in feat_cols if c not in df_blind.columns]
        if missing:
            log(f"[ERROR] Missing columns: {missing} - skipping.", prefix="  !!")
            continue

        # Impute NaN with context means
        X = df_blind[feat_cols].values.astype(np.float32)
        X[np.isinf(X)] = np.nan
        mask = np.isnan(X)
        X[mask] = np.take(nan_means, np.where(mask)[1])

        # Generate probability vectors in batches
        BATCH = config.EMBED_BATCH_SIZE
        if len(X) <= BATCH:
            embeddings = model.predict_proba(X).astype(np.float32)
        else:
            parts = []
            for i in range(0, len(X), BATCH):
                parts.append(model.predict_proba(X[i:i+BATCH]).astype(np.float32))
            embeddings = np.vstack(parts)

        # Save probability vectors to TXT
        txt_path = os.path.join(paths["embeddings_dir"],
                                f"embeddings_{true_class}.txt")
        save_embeddings_txt(true_class, embeddings, all_classes, txt_path)
        log(f"Probability vectors : {txt_path}")

        # Prediction with balancing
        embeddings_bal = embeddings * class_weights
        argmax_bal     = np.argmax(embeddings_bal, axis=1)
        y_pred         = np.array([all_classes[i] for i in argmax_bal])

        # Distribution table
        N = len(y_pred)
        print(f"\n  {'Pos':<5}  {'Class':<10}  {'Count':>8}  {'%':>8}  "
              f"{'Bar':<28}  {'':^13}", flush=True)
        print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  "
              f"{'-'*28}  {'-'*13}", flush=True)
        for pos, cls in enumerate(all_classes):
            cnt     = int(np.sum(argmax_bal == pos))
            pct     = cnt / N * 100
            bar     = "█" * int(pct / 2) + "░" * (28 - int(pct / 2))
            is_real = "<- CORRECT [v]" if cls == true_class else ""
            print(f"  {pos:<5}  {cls:<10}  {cnt:>8,}  {pct:>7.2f}%  "
                  f"{bar:<28}  {is_real}", flush=True)

        # Accumulate for confusion matrix
        if true_class not in cm_dict:
            cm_dict[true_class] = {}
        for pred in y_pred:
            cm_dict[true_class][pred] = cm_dict[true_class].get(pred, 0) + 1

    # -- Global metrics ---------------------------------------------------------
    sep(f"RESULTS PARTITION {fold}", char="=")

    known_seen = sorted(
        set(cm_dict.keys()) |
        {p for v in cm_dict.values() for p in v.keys()}
    )
    cm_np, cm_classes, accuracy, per_class, f1_macro, f1_weighted = \
        compute_metrics(cm_dict, known_seen)

    print(f"\n  {'Class':<10}  {'Support':>8}  {'Precision':>10}  "
          f"{'Recall':>8}  {'F1':>8}", flush=True)
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}", flush=True)
    for cls in cm_classes:
        m = per_class[cls]
        print(f"  {cls:<10}  {m['support']:>8,}  "
              f"{m['precision']:>9.3f}  "
              f"{m['recall']:>8.3f}  "
              f"{m['f1']:>8.3f}", flush=True)
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}", flush=True)
    print(f"\n  Global Accuracy : {accuracy*100:.2f}%")
    print(f"  F1 Macro        : {f1_macro*100:.2f}%")
    print(f"  F1 Weighted     : {f1_weighted*100:.2f}%", flush=True)

    # Save confusion matrix and CSV
    save_confusion_matrix(cm_np, cm_classes, paths["confusion_matrix"], fold)
    log(f"Confusion matrix : {paths['confusion_matrix']}")

    rows = []
    for cls in cm_classes:
        m = per_class[cls]
        rows.append({"partition": fold, "clase": cls,
                     "support": m["support"],
                     "precision": round(m["precision"], 4),
                     "recall":    round(m["recall"],    4),
                     "f1":        round(m["f1"],        4),
                     "TP": m["TP"], "FP": m["FP"], "FN": m["FN"]})
    rows.append({"partition": fold, "clase": "GLOBAL",
                 "support": sum(per_class[c]["support"] for c in cm_classes),
                 "precision": round(float(np.mean([per_class[c]["precision"] for c in cm_classes])), 4),
                 "recall":    round(float(np.mean([per_class[c]["recall"] for c in cm_classes])), 4),
                 "f1":        round(f1_macro, 4),
                 "TP": "", "FP": "", "FN": ""})
    pd.DataFrame(rows).to_csv(paths["metricas_csv"], index=False)


    return {"accuracy": accuracy, "f1_macro": f1_macro,
            "f1_weighted": f1_weighted, "per_class": per_class,
            "classes": cm_classes}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TabPFN Evaluation per partition (includes context fitting)"
    )
    parser.add_argument("--partition", type=int, required=True,
                        choices=range(1, config.N_PARTITIONS + 1),
                        help="Partition to evaluate (1, 2, 3 or 4)")
    args = parser.parse_args()
    fold = args.partition

    np.random.seed(config.RANDOM_SEED)
    start = time.time()

    W = 58
    print(f"\n  {'='*W}")
    print(f"  EVALUATION - PARTITION {fold} / {config.N_PARTITIONS}")
    print(f"  TabPFN argmax + balance alpha={config.CLASS_WEIGHT_ALPHA}")
    print(f"  {'='*W}\n")


    evaluar_fold(fold)

    elapsed = time.time() - start

    print(f"\n  {'='*W}")
    print(f"  Partition {fold} completed in {elapsed/60:.1f} min")
    print(f"  Results in: {config.get_partition_paths(fold)['results_dir']}")
    print(f"  {'='*W}\n")



if __name__ == "__main__":
    main()