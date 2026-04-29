import os
import re
import numpy as np
from datetime import date
from obspy import read as mseed_read, Trace, Stream
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Input directories
SISMOS_ROOT  = r"C:\Users\Daniel\OneDrive\Escritorio\Sismos"
OUTPUT_DIR   = r"C:\Users\Daniel\OneDrive\Escritorio\Sismos\TOR"

# Sensor parameters
ESTACION     = "ANGP"
RED          = "CM"
CANAL        = "ELZ"

# Target number of augmented tornillos required
TARGET_TOTAL = 3180 

# ── Alpha bounds for each augmentation category (low / medium / high) ────────
ALPHA_BANDS = {
    "low":  (0.35, 0.50),
    "medium": (0.50, 0.65),
    "high":  (0.65, 0.80),
}

# ── Relative energy control bounds ────────────────────────────────────────────
ENERGY_RATIO_MIN = 0.05
ENERGY_RATIO_MAX = 0.80

# ── Noise vector construction ─────────────────────────────────────────────────
MIN_NOISE_SAMPLES = 300
MAX_SOURCES       = 50
DIAS_PERIODO      = 7
NOISE_CLASSES     = ["VT", "LP", "TRE"]

RANDOM_SEED  = 42

# ==============================================================================
# DATE UTILITIES
# ==============================================================================

def parse_event_date(evt_folder: str, year_str: str):
    m = re.match(r'^(\d{2})(\d{2})\d{6}', evt_folder)
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    try:
        return date(2000 + int(year_str), month, day)
    except (ValueError, OverflowError):
        return None


def days_apart(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    return abs((a - b).days)

# ==============================================================================
# 1. DISCOVERY
# ==============================================================================

def _match_file(filename: str) -> bool:
    if not filename.lower().endswith((".mseed", ".miniseed", ".ms")):
        return False
    up = filename.upper()
    return ESTACION in up and CANAL in up


def discover_tornillos() -> list:
    """Returns list of (filepath, year_str, evt_folder, event_date)."""
    to_dir  = os.path.join(SISMOS_ROOT, "TOR")
    results = []
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
                    results.append((
                        os.path.join(evt_path, f),
                        yr, evt,
                        parse_event_date(evt, yr),
                    ))
                    break
    return results


def discover_noise_sources(years: list) -> list:
    """Returns list of (filepath, event_date) for LP/TR/VA from the same years."""
    results = []
    for cls in NOISE_CLASSES:
        cls_dir = os.path.join(SISMOS_ROOT, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARNING] Folder not found: {cls_dir}")
            continue
        for yr in years:
            yr_path = os.path.join(cls_dir, yr)
            if not os.path.isdir(yr_path):
                print(f"  [WARNING] Not found {cls}/{yr} — skipping")
                continue
            for evt in sorted(os.listdir(yr_path)):
                evt_path = os.path.join(yr_path, evt)
                if not os.path.isdir(evt_path):
                    continue
                for f in os.listdir(evt_path):
                    if _match_file(f):
                        results.append((
                            os.path.join(evt_path, f),
                            parse_event_date(evt, yr),
                        ))
                        break
    return results

# ==============================================================================
# 2. READ AND NORMALIZE TORNILLO
# ==============================================================================

def read_and_normalize(filepath: str):
    """
    Returns (data_norm, stats, scale_ref) or (None, None, None).
    scale_ref = max(|x_raw|) — used later to rescale the synthetic signal.
    """
    try:
        tr        = mseed_read(filepath)[0]
        data      = tr.data.astype(np.float64)
        scale_ref = np.max(np.abs(data))
        if scale_ref == 0:
            return None, None, None
        return data / scale_ref, tr.stats, scale_ref
    except Exception:
        return None, None, None

# ==============================================================================
# 3. EXTRACT REAL PREAMBLE
# ==============================================================================

def _detect_onset(data: np.ndarray, win: int = 200, thresh: float = 2.0) -> int:
    n     = len(data)
    n_ref = 5
    if n < win * (n_ref + 2):
        return int(n * 0.10)
    n_win    = n // win
    energies = np.array([
        np.sqrt(np.mean(data[i * win : (i + 1) * win] ** 2))
        for i in range(n_win)
    ])
    base = np.mean(energies[:n_ref])
    if base == 0:
        return int(n * 0.10)
    for i in range(n_ref, n_win):
        if energies[i] > base * thresh:
            return i * win
    return int(n * 0.10)


def _extract_preamble(filepath: str, max_samples: int):
    """
    Extracts real noise preamble from a signal (without normalizing).

    Returns an array of up to max_samples, or None if:
      - the preamble is smaller than MIN_NOISE_SAMPLES, or
      - fails flat noise validation (peak/RMS > 3.5).
    """
    try:
        tr        = mseed_read(filepath)[0]
        data      = tr.data.astype(np.float64)
        onset     = _detect_onset(data)
        max_noise = int(len(data) * 0.20)
        onset     = min(onset, max_noise)
        noise_raw = data[:onset]

        if len(noise_raw) < MIN_NOISE_SAMPLES:
            return None

        rms  = np.sqrt(np.mean(noise_raw ** 2))
        peak = np.max(np.abs(noise_raw))
        if rms > 0 and peak > 3.5 * rms:
            return None

        return noise_raw[:max_samples]

    except Exception:
        return None

# ==============================================================================
# 4. BUILD NOISE VECTOR BY CONCATENATION
# ==============================================================================

def build_noise_vector(candidates: list, target_length: int) -> np.ndarray | None:
    """
    Builds a noise vector of exactly target_length samples
    by concatenating real preambles from different sources.

    The result is 100% real: no tiling, no artificial zeros.
    Each chunk comes from a different source (never repeated).
    """
    chunks       = []
    collected    = 0
    sources_used = 0

    for fp, _ in candidates:
        if collected >= target_length:
            break
        if sources_used >= MAX_SOURCES:
            break

        needed   = target_length - collected
        preamble = _extract_preamble(fp, max_samples=needed)
        if preamble is None or len(preamble) == 0:
            continue

        chunks.append(preamble)
        collected    += len(preamble)
        sources_used += 1

    if collected < target_length:
        return None

    noise_raw = np.concatenate(chunks)[:target_length]
    mx = np.max(np.abs(noise_raw))
    if mx == 0:
        return None
    return noise_raw / mx

# ==============================================================================
# 5. NOISE SELECTION AND QUOTAS PER BAND
# ==============================================================================

def _alpha_valid_range(tornillo_norm: np.ndarray, noise_norm: np.ndarray) -> tuple:
    """Alpha range such that RMS energy ratio falls in [ENERGY_RATIO_MIN, ENERGY_RATIO_MAX]."""
    rms_t = np.sqrt(np.mean(tornillo_norm ** 2))
    rms_n = np.sqrt(np.mean(noise_norm ** 2))
    if rms_t == 0 or rms_n == 0:
        return (ALPHA_BANDS["bajo"][0], ALPHA_BANDS["alto"][1])
    return (ENERGY_RATIO_MIN * rms_t / rms_n,
            ENERGY_RATIO_MAX * rms_t / rms_n)


def pick_noise_and_alpha(noise_pool: list, tornillo_norm: np.ndarray,
                         tornillo_date, count_band: dict,
                         band_target: dict, rng) -> tuple:
    """
    Builds the noise vector and selects α respecting quotas per band.

    Logic:
      1. Filters bands that reached quota → only works with open bands.
      2. Orders open bands by count_band ascending (most lagging first).
      3. Uses the first band compatible with energy range [e_lo, e_hi].
      4. If no open band is compatible → None, None, None.

    Returns (noise_vector, alpha, band_name) or (None, None, None).
    """
    close = [(fp, d) for fp, d in noise_pool if days_apart(d, tornillo_date) <= DIAS_PERIODO]
    far   = [(fp, d) for fp, d in noise_pool if days_apart(d, tornillo_date) >  DIAS_PERIODO]
    close = list(close); rng.shuffle(close)
    far   = list(far);   rng.shuffle(far)
    candidates = close + far

    noise = build_noise_vector(candidates, len(tornillo_norm))
    if noise is None:
        return None, None, None

    e_lo, e_hi = _alpha_valid_range(tornillo_norm, noise)

    # Only bands with pending quota, ordered by the most lagging first
    remaining_bands = [b for b in ALPHA_BANDS if count_band[b] < band_target[b]]
    band_order = sorted(remaining_bands, key=lambda b: count_band[b])

    for band in band_order:
        b_lo, b_hi = ALPHA_BANDS[band]
        lo = max(b_lo, e_lo)
        hi = min(b_hi, e_hi)
        if lo <= hi:
            alpha = float(rng.uniform(lo, hi))
            return noise, alpha, band

    return None, None, None

# ==============================================================================
# 6. AUGMENTATION AND SAVING
# ==============================================================================

def augment_tornillo(tornillo_norm: np.ndarray,
                     noise_norm: np.ndarray,
                     alpha: float) -> np.ndarray:
    """
    Mix in normalized space:
        x_mix = x_tornillo_norm + α · x_noise_norm
    """
    return tornillo_norm + noise_norm * alpha


def build_output_path(original_stats, alpha: float,
                      yr_str: str, evt_folder: str) -> tuple:
    """
    Path compatible with real dataset:
      to/{year}_Aug/{evt}_a{alpha_int:03d}/ANGP.CM.20.ELZ.{MMDDHHMMSS}_estaciones.mseed
    """
    t           = original_stats.starttime
    time_tag    = f"{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}"
    alpha_int   = round(alpha * 1000)
    folder_name = f"{evt_folder}_a{alpha_int:03d}"
    out_dir     = os.path.join(OUTPUT_DIR, f"{yr_str}_Aug", folder_name)
    filename    = f"ANGP.CM.20.ELZ.{time_tag}_a{alpha_int:03d}_estaciones.mseed"
    return os.path.join(out_dir, filename), out_dir


def save_augmented(data: np.ndarray, original_stats, output_path: str):
    tr = Trace(data=data.astype(np.float32))
    tr.stats.station       = original_stats.station
    tr.stats.network       = original_stats.network
    tr.stats.channel       = original_stats.channel
    tr.stats.location      = original_stats.location
    tr.stats.sampling_rate = original_stats.sampling_rate
    tr.stats.starttime     = original_stats.starttime
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Stream([tr]).write(output_path, format="MSEED")

# ==============================================================================
# 7. MAIN PIPELINE
# ==============================================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    W = 64
    print(f"\n  {'═'*W}")
    print(f"  DATA AUGMENTATION — TORNILLOS  (v9 — global limits)")
    print(f"  {'═'*W}")
    print(f"  Sismos root  : {SISMOS_ROOT}")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"  Station      : {ESTACION} | Network: {RED} | Channel: {CANAL}")
    print(f"  Alpha Bands  : low {ALPHA_BANDS['low']}  "
          f"medium {ALPHA_BANDS['medium']}  high {ALPHA_BANDS['high']}")
    print(f"  Energy Ratio : [{ENERGY_RATIO_MIN}, {ENERGY_RATIO_MAX}]")
    print(f"  Scale        : x_final = x_mix · A_ref")
    print(f"  Noise        : concatenated real preambles")
    print(f"  Max sources  : {MAX_SOURCES} per noise vector")
    print(f"  Target total : {TARGET_TOTAL}")
    print(f"  {'═'*W}\n")

    # ── [1] Tornillos ────────────────────────────────────────────────────────
    print("[1] Discovering real tornillos...")
    tornillos = discover_tornillos()
    if not tornillos:
        print("[ERROR] No tornillos found.")
        return
    years = sorted(set(yr for _, yr, _, _ in tornillos))
    print(f"  Found : {len(tornillos)}")
    for yr in years:
        cnt = sum(1 for _, y, _, _ in tornillos if y == yr)
        print(f"    {yr} → {cnt} tornillos")

    # ── [2] Noise sources ────────────────────────────────────────────────────
    print("\n[2] Searching for noise sources (LP / TR / VA)...")
    noise_pool = discover_noise_sources(years)
    print(f"  Sources found: {len(noise_pool)}")
    if len(noise_pool) < 10:
        print("[ERROR] Too few noise sources.")
        return

    # ── [3] Read tornillos ───────────────────────────────────────────────────
    print("\n[3] Reading and normalizing tornillos...")
    tornillo_data = []
    for fp, yr, evt, evt_date in tqdm(tornillos, desc="  Reading"):
        data_norm, stats, scale_ref = read_and_normalize(fp)
        if data_norm is not None and len(data_norm) > 500:
            tornillo_data.append((data_norm, stats, yr, evt, evt_date, scale_ref))

    n_reales     = len(tornillo_data)
    n_sinteticos = TARGET_TOTAL - n_reales
    print(f"  Valid              : {n_reales}")
    print(f"  Synthetics needed  : {n_sinteticos}")

    if n_sinteticos <= 0:
        print(f"  Already have {n_reales} ≥ {TARGET_TOTAL}. Augmentation not needed.")
        return

    # ── [4] Quotas per band ──────────────────────────────────────────────────
    target_per_band = n_sinteticos // 3
    band_names      = list(ALPHA_BANDS.keys())   # ["low", "medium", "high"]
    band_target     = {b: target_per_band for b in band_names}
    for i in range(n_sinteticos % 3):            # distribute remainder to first bands
        band_target[band_names[i]] += 1

    count_band = {b: 0 for b in band_names}

    print(f"\n[4] Quotas per band:")
    for b in band_names:
        print(f"  {b:>5}  α ∈ {ALPHA_BANDS[b]}  →  {band_target[b]} synthetics")

    # ── [5] Generation ───────────────────────────────────────────────────────
    print(f"\n[5] Generating augmented tornillos...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    count_ok            = 0
    count_sin_ruido     = 0
    alphas_used         = []
    order               = list(range(n_reales))
    max_global_attempts = 5 * n_sinteticos
    attempts            = 0

    pbar = tqdm(total=n_sinteticos, desc="  Generating")

    while count_ok < n_sinteticos:
        rng.shuffle(order)

        for idx in order:
            if count_ok >= n_sinteticos:
                break
            if attempts >= max_global_attempts:
                break

            # Exit if all quotas are complete
            if not any(count_band[b] < band_target[b] for b in band_names):
                break

            attempts += 1
            t_data, t_stats, t_yr, t_evt, t_date, t_scale = tornillo_data[idx]

            noise, alpha, band = pick_noise_and_alpha(
                noise_pool, t_data, t_date, count_band, band_target, rng
            )

            if noise is None:
                count_sin_ruido += 1
                continue

            augmented_norm = augment_tornillo(t_data, noise, alpha)
            augmented_raw  = augmented_norm * t_scale

            out_path, _ = build_output_path(t_stats, alpha, t_yr, t_evt)
            save_augmented(augmented_raw, t_stats, out_path)

            count_band[band] += 1
            alphas_used.append(alpha)
            count_ok += 1
            pbar.update(1)

        if attempts >= max_global_attempts:
            print(f"\n  [WARNING] Reached attempt limit {max_global_attempts:,} "
                  f"with {count_ok:,} synthetics generated.")
            break

    pbar.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    alphas_arr = np.array(alphas_used) if alphas_used else np.array([0.0])
    print(f"\n  {'═'*W}")
    print(f"  SUMMARY")
    print(f"  {'═'*W}")
    print(f"  Real tornillos       : {n_reales:,}")
    print(f"  Generated synthetics : {count_ok:,}")
    print(f"  TOTAL                : {n_reales + count_ok:,}")
    print(f"  Total attempts       : {attempts:,}  (limit: {max_global_attempts:,})")
    print(f"\n  Distribution by band:")
    for b in band_names:
        pct = 100 * count_band[b] / count_ok if count_ok else 0
        print(f"    {b:>5}  α ∈ {ALPHA_BANDS[b]}  →  {count_band[b]:,}  ({pct:.1f} %)")
    if count_sin_ruido:
        pct = 100 * count_sin_ruido / (count_ok + count_sin_ruido) if (count_ok + count_sin_ruido) > 0 else 0
        print(f"\n  Skipped (no noise found) : {count_sin_ruido:,}  ({pct:.1f} %)")
    print(f"\n  Alphas used:")
    print(f"    min  = {alphas_arr.min():.4f}")
    print(f"    p25  = {np.percentile(alphas_arr, 25):.4f}")
    print(f"    med  = {np.median(alphas_arr):.4f}")
    print(f"    p75  = {np.percentile(alphas_arr, 75):.4f}")
    print(f"    max  = {alphas_arr.max():.4f}")
    print(f"\n  Output structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  └── {{year}}_Aug/")
    print(f"      └── {{event}}_a{{alpha_int}}/")
    print(f"          └── ANGP.CM.20.ELZ.{{MMDDHHMMSS}}_a{{alpha_int}}_estaciones.mseed")
    print(f"  {'═'*W}\n")


if __name__ == "__main__":
    main()
