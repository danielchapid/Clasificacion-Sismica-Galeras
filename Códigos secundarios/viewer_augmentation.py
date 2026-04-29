import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import read as mseed_read
from scipy.signal import welch

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input directories
SISMOS_ROOT   = # Dataset access path
AUGMENTED_DIR = # Path to augmented data

# Output directory for generated figures
OUTPUT_FIGS   = r"C:\Users\Daniel\OneDrive\Documentos\Tesis_Galeras_Final\Data augmentation\figuras_v5"

ESTACION = "ANGP"
CANAL    = "ELZ"

# Figures to generate per range
N_POR_RANGO    = 35      
ALPHA_LOW_MAX  = 0.50    # α < 0.50  → bajo
ALPHA_HIGH_MIN = 0.65    # α ≥ 0.65  → alto

HIGHPASS_FREQ = 0.7      # highpass cutoff frequency [Hz]

FIG_DPI = 150

COLOR_ORIG = "#072F5F"   # dark blue
COLOR_AUG  = "#D84315"   # dark red (rust)

RANDOM_SEED = 42         # None for pure random on each run

# ==============================================================================
# READING AND INDEXING
# ==============================================================================

def _match_file(filename: str) -> bool:
    up = filename.upper()
    return (ESTACION in up and CANAL in up
            and filename.lower().endswith((".mseed", ".miniseed", ".ms")))


def read_signal(filepath: str):
    """Returns (float64 data, sampling_rate) with preprocessing:
       offset removal (detrend constant + linear) and zero-phase 
       highpass filter at HIGHPASS_FREQ Hz."""
    tr = mseed_read(filepath)[0]
    tr.detrend("constant")
    tr.detrend("linear")
    try:
        tr.filter("highpass", freq=HIGHPASS_FREQ, zerophase=True)
    except Exception:
        tr.filter("highpass", freq=HIGHPASS_FREQ, zerophase=False)
    return tr.data.astype(np.float64), float(tr.stats.sampling_rate)


def index_originals() -> dict:
    """Returns {evt_folder: filepath}."""
    to_dir = os.path.join(SISMOS_ROOT, "TOR")
    result = {}
    for yr in sorted(os.listdir(to_dir)):
        yr_path = os.path.join(to_dir, yr)
        if not os.path.isdir(yr_path):
            continue
        for evt in sorted(os.listdir(yr_path)):
            evt_path = os.path.join(yr_path, evt)
            if not os.path.isdir(evt_path):
                continue
            for f in os.listdir(evt_path):
                if _match_file(f):
                    result[evt] = os.path.join(evt_path, f)
                    break
    return result


def index_augmented() -> dict:
    """
    Returns {alpha_int: [(aug_path, evt_folder, alpha_float), ...]}.

    Expected folder format (augmenter v9 naming):
        {evt_folder}_a{alpha_int:03d}
    Example: 0512143022_a312
    """
    groups  = {}
    # 3 digits for alpha (resolution 0.001)
    pattern = re.compile(r'^(.+)_a(\d{3})$')
    for yr in sorted(os.listdir(AUGMENTED_DIR)):
        if not yr.endswith("_Aug"):
            continue
        yr_path = os.path.join(AUGMENTED_DIR, yr)
        if not os.path.isdir(yr_path):
            continue
        for folder in sorted(os.listdir(yr_path)):
            m = pattern.match(folder)
            if not m:
                continue
            evt_folder = m.group(1)
            alpha_int  = int(m.group(2))   # ej. 312
            folder_path = os.path.join(yr_path, folder)
            for f in os.listdir(folder_path):
                if _match_file(f):
                    groups.setdefault(alpha_int, []).append(
                        (os.path.join(folder_path, f),
                         evt_folder,
                         alpha_int / 1_000.0)  # 3 decimales
                    )
                    break
    return groups

# ==============================================================================
# GROUPING BY RANGES
# ==============================================================================

def build_ranges(aug_groups: dict) -> dict:
    rangos = {"low": [], "medium": [], "high": []}
    for alpha_int, items in aug_groups.items():
        a = alpha_int / 1_000.0
        if a < ALPHA_LOW_MAX:
            rangos["low"].extend(items)
        elif a < ALPHA_HIGH_MIN:
            rangos["medium"].extend(items)
        else:
            rangos["high"].extend(items)
    return rangos


def sample_range(items: list, n: int, rng: np.random.Generator) -> list:
    """Selects n random items without replacement."""
    if len(items) <= n:
        return items
    indices = rng.choice(len(items), size=n, replace=False)
    return [items[i] for i in indices]

# ==============================================================================
# FIGURE: 3 PANELS
# ==============================================================================

def plot_three_panels(orig_full: np.ndarray, aug_full: np.ndarray,
                      sr: float, alpha: float, evt_name: str,
                      rango: str, out_path: str):
    """
    Panel 1: Synthetic waveform (augmented).
    Panel 2: Original waveform.
    Panel 3: Welch PSD — original vs augmented overlaid.
    """
    # ── Prepare signals ──────────────────────────────────────────────────────
    n    = min(len(orig_full), len(aug_full))
    orig = orig_full[:n]
    aug  = aug_full [:n]
    t    = np.arange(n) / sr

    # Individual normalization (each signal to its own absolute maximum)
    scale_orig = np.max(np.abs(orig))
    scale_aug  = np.max(np.abs(aug))
    if scale_orig == 0:
        scale_orig = 1.0
    if scale_aug == 0:
        scale_aug = 1.0
    orig_n = orig / scale_orig
    aug_n  = aug  / scale_aug

    # ── PSD ───────────────────────────────────────────────────────────────────
    nperseg  = max(256, min(1024, n // 4))
    f_o, p_o = welch(orig, fs=sr, nperseg=nperseg)
    f_a, p_a = welch(aug,  fs=sr, nperseg=nperseg)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 11),
        gridspec_kw={"hspace": 0.50, "height_ratios": [1, 1, 1]}
    )

    fig.suptitle(
        f"Station: {ESTACION} | Channel: {CANAL}",
        fontsize=15, fontweight="bold", y=0.96
    )

    # ── Panel 1: Synthetic waveform ──────────────────────────
    ax = axes[0]
    ax.plot(t, aug_n, color=COLOR_AUG, linewidth=1.2, alpha=0.9,
            label=f"Synthetic ($\\alpha$={alpha:.3f})")
    ax.set_title("Synthetic waveform", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Normalized Amplitude", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.25, 1.25)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.75)
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.tick_params(labelsize=10)

    # ── Panel 2: Original waveform ───────────────────────────────────────
    ax = axes[1]
    ax.plot(t, orig_n, color=COLOR_ORIG, linewidth=1.2, alpha=0.9,
            label="Original")
    ax.set_title("Original waveform", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Normalized Amplitude", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.25, 1.25)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.75)
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.tick_params(labelsize=10)

    # ── Panel 3: Overlaid PSD ──────────────────────────────────────────────
    ax = axes[2]
    ax.semilogy(f_o, p_o, color=COLOR_ORIG, linewidth=1.2, alpha=0.9,
                label="Original")
    ax.semilogy(f_a, p_a, color=COLOR_AUG, linewidth=1.2, alpha=0.8,
                label=f"Synthetic ($\\alpha$={alpha:.3f})")
    ax.set_title("Power Spectrum Density (PSD)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency [Hz]", fontsize=12)
    ax.set_ylabel("PSD", fontsize=12)
    ax.set_xlim(0, 50)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.75)
    ax.grid(True, alpha=0.22, linewidth=0.5, which="both")
    ax.tick_params(labelsize=10)

    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"      ✓ {os.path.basename(out_path)}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"\n  {'='*66}")
    print(f"  VIEW AUGMENTED TORNILLOS  (v5 — naming v9)")
    print(f"  {'='*66}")
    print(f"  Ranges : low < {ALPHA_LOW_MAX}  |  medium [{ALPHA_LOW_MAX}, {ALPHA_HIGH_MIN})  |  high >= {ALPHA_HIGH_MIN}")
    print(f"  Seed   : {RANDOM_SEED}  ({'reproducible' if RANDOM_SEED is not None else 'pure random'})")
    print(f"  Output : {OUTPUT_FIGS}\n")

    print("  [1] Indexing originals...")
    originals = index_originals()
    print(f"      {len(originals)} found")

    print("\n  [2] Indexing augmented...")
    if not os.path.isdir(AUGMENTED_DIR):
        print(f"[ERROR] Folder does not exist. Did you run data_augmentation_tornillo.py?")
        print(f"  {AUGMENTED_DIR}")
        return
    aug_groups = index_augmented()
    if not aug_groups:
        print(f"[ERROR] No augmented files found in:\n  {AUGMENTED_DIR}")
        return
    total_aug = sum(len(v) for v in aug_groups.values())
    print(f"      {total_aug} augmented  |  {len(aug_groups)} alpha values")

    print("\n  [3] Classifying by ranges...")
    rangos = build_ranges(aug_groups)
    for nombre, items in rangos.items():
        print(f"      {nombre:5s} : {len(items):4d} augmented")

    print(f"\n  [4] Generating figures ({N_POR_RANGO} per range, random selection)...")
    os.makedirs(OUTPUT_FIGS, exist_ok=True)

    total_figs = 0
    skipped    = 0

    for rango_nombre in ["low", "medium", "high"]:
        items = rangos[rango_nombre]
        if not items:
            print(f"\n  [WARNING] Range '{rango_nombre}' is empty — skipping")
            continue

        selected = sample_range(items, N_POR_RANGO, rng)
        out_dir  = os.path.join(OUTPUT_FIGS, rango_nombre)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  Range {rango_nombre.upper()}  ({len(selected)} example(s)):")

        for j, (aug_path, evt_folder, alpha_f) in enumerate(selected):
            if evt_folder not in originals:
                print(f"      [WARNING] Original not found for '{evt_folder}'")
                skipped += 1
                continue
            try:
                orig_data, sr = read_signal(originals[evt_folder])
                aug_data,  _  = read_signal(aug_path)
            except Exception as e:
                print(f"      [ERROR] {evt_folder}: {e}")
                skipped += 1
                continue

            out_path = os.path.join(
                out_dir,
                f"{evt_folder}_a{round(alpha_f*1_000):03d}.png"
            )
            plot_three_panels(orig_data, aug_data, sr, alpha_f,
                            evt_folder, rango_nombre, out_path)
            total_figs += 1

    print(f"\n  {'='*66}")
    print(f"  {total_figs} figures generated")
    if skipped:
        print(f"  {skipped} skipped (original not found or read error)")
    print(f"\n  Output structure:")
    print(f"  {OUTPUT_FIGS}/")
    print(f"  ├── low/    (alpha < {ALPHA_LOW_MAX})")
    print(f"  ├── medium/ ({ALPHA_LOW_MAX} <= alpha < {ALPHA_HIGH_MIN})")
    print(f"  └── high/   (alpha >= {ALPHA_HIGH_MIN})")
    print(f"  {'='*66}\n")


if __name__ == "__main__":
    main()
