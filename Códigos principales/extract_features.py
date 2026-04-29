import sys
sys.dont_write_bytecode = True
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import librosa
from obspy import read as mseed_read
from scipy.signal import welch, find_peaks, hilbert
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config_extract as cfg

warnings.filterwarnings("ignore")


def base_event_id(event_id: str) -> str:
    return re.sub(r'_a\d+$', '', event_id)


def discover_events(input_root: str) -> dict:
    class_events = {}
    extensions   = (".mseed", ".miniseed", ".ms")

    for class_name in sorted(os.listdir(input_root)):
        class_dir = os.path.join(input_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        events = []
        for root_dir, _, filenames in os.walk(class_dir):
            if any(f.lower().endswith(extensions) for f in filenames):
                events.append(root_dir)
        if events:
            class_events[class_name] = sorted(events)

    return class_events


def sample_events(class_events: dict) -> dict:
    rng = np.random.default_rng(cfg.PARTITION_SEEDS[0])
    sampled = {}
    max_ev = getattr(cfg, 'MAX_TRZ', None)
    if max_ev is not None:
        total_budget = max_ev
        sorted_classes = sorted(class_events.keys(), key=lambda k: len(class_events[k]))
        per_class_allocation = {}
        for i, cls in enumerate(sorted_classes):
            remaining_classes = len(sorted_classes) - i
            fair_share = int(total_budget / remaining_classes)
            available = len(class_events[cls])
            if available < fair_share:
                assigned = available
            else:
                assigned = fair_share
            per_class_allocation[cls] = assigned
            total_budget -= assigned
        for cls, events in class_events.items():
            n = per_class_allocation[cls]
            if n >= len(events):
                sampled[cls] = events[:]
            else:
                idx = rng.choice(len(events), size=n, replace=False)
                sampled[cls] = [events[i] for i in sorted(idx)]
        return sampled

    n = getattr(cfg, 'EVENTS_PER_CLASS', None)
    if n is None:
        return {cls: evs[:] for cls, evs in class_events.items()}

    for cls, events in class_events.items():
        if len(events) > n:
            idx = rng.choice(len(events), size=n, replace=False)
            sampled[cls] = [events[i] for i in sorted(idx)]
        else:
            sampled[cls] = events[:]
    return sampled


# Preprocessing
def preprocess_trace(trace):
    tr = trace.copy()
    tr.detrend("constant")
    tr.detrend("linear")
    try:
        tr.filter("highpass", freq=cfg.HIGHPASS_FREQ, zerophase=True)
    except Exception:
        tr.filter("highpass", freq=cfg.HIGHPASS_FREQ, zerophase=False)
    if abs(tr.stats.sampling_rate - cfg.TARGET_SAMPLERATE) > 0.1:
        tr.resample(cfg.TARGET_SAMPLERATE)
    return tr.data.astype(np.float64), float(tr.stats.sampling_rate)


# Features

def features_time_domain(signal: np.ndarray, sr: float) -> dict:
    n            = len(signal)
    duration     = n / sr
    total_energy = float(np.sum(signal ** 2))
    max_abs      = float(np.max(np.abs(signal)))
    rms_val      = float(np.sqrt(np.mean(signal ** 2)))

    normalized_total_energy = (
        total_energy / (max_abs ** 2 * n) if (max_abs > 0 and n > 0) else 0.0
    )
    crest_factor = (max_abs / rms_val) if rms_val > 0 else 0.0

    analytic_signal = hilbert(signal)
    envelope        = np.abs(analytic_signal)

    zcr          = len(np.where(np.diff(np.sign(signal)))[0]) / duration if duration > 0 else 0.0
    skewness_val = float(skew(signal))
    kurtosis_val = float(kurtosis(signal))

    peak_idx       = int(np.argmax(envelope))
    coda_threshold = cfg.CODA_THRESHOLD * max_abs
    post_peak_env  = envelope[peak_idx:]
    below          = np.where(post_peak_env < coda_threshold)[0]
    noise_end      = max(1, int(n * cfg.NOISE_WINDOW_FRACTION))
    noise_rms      = float(np.sqrt(np.mean(signal[:noise_end] ** 2))) + 1e-12
    snr_check      = 20.0 * np.log10((rms_val + 1e-12) / noise_rms)

    if snr_check < cfg.SNR_MIN_FOR_CODA:
        coda_duration = float("nan")
    elif len(below) > 0:
        coda_duration = float(below[0] / sr)
    else:
        coda_duration = float((n - peak_idx) / sr)

    half                    = n // 2
    e_first_half            = float(np.sum(signal[:half] ** 2))
    energy_ratio_first_half = (e_first_half / total_energy) if total_energy > 0 else 0.5

    envelope_skewness = float(skew(envelope))

    return {
        "duration"                : duration,
        "total_energy"            : total_energy,
        "normalized_total_energy" : normalized_total_energy,
        "peak_amplitude"          : max_abs,
        "rms"                     : rms_val,
        "crest_factor"            : crest_factor,
        "zero_crossing_rate"      : zcr,
        "skewness"                : skewness_val,
        "kurtosis"                : kurtosis_val,
        "coda_duration"           : coda_duration,
        "energy_ratio_first_half" : energy_ratio_first_half,
        "envelope_skewness"       : envelope_skewness,
    }


def features_freq_domain(signal: np.ndarray, sr: float) -> dict:
    nperseg    = min(cfg.WELCH_NPERSEG, len(signal))
    freqs, psd = welch(signal, fs=sr, nperseg=nperseg)

    psd_total = np.sum(psd) or 1e-12
    psd_norm  = psd / psd_total

    dominant_freq = float(freqs[np.argmax(psd)])
    central_freq  = float(np.sum(freqs * psd_norm))

    cumulative     = np.cumsum(psd_norm)
    idx_low        = np.searchsorted(cumulative, (1.0 - cfg.ENERGY_BAND_PCT) / 2.0)
    idx_high       = np.searchsorted(cumulative, 1.0 - (1.0 - cfg.ENERGY_BAND_PCT) / 2.0)
    energy_band_70 = float(freqs[idx_high] - freqs[idx_low]) if idx_high < len(freqs) else 0.0

    spectral_entropy  = float(-np.sum((psd_norm + 1e-12) * np.log2(psd_norm + 1e-12)))
    spectral_flatness = float(
        np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)
    )

    cumulative_abs   = np.cumsum(psd)
    rolloff_idx      = np.searchsorted(cumulative_abs, cfg.SPECTRAL_ROLLOFF * cumulative_abs[-1])
    spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    peak_count = int(len(find_peaks(psd, height=0.10 * np.max(psd))[0]))

    mask_low  = (freqs >= cfg.SPECRATIO_LOW_MIN)  & (freqs < cfg.SPECRATIO_LOW_MAX)
    mask_high = (freqs >= cfg.SPECRATIO_HIGH_MIN) & (freqs < cfg.SPECRATIO_HIGH_MAX)
    e_low     = float(np.sum(psd[mask_low]))  + 1e-12
    e_high    = float(np.sum(psd[mask_high])) + 1e-12
    spectral_ratio_low_high  = e_low / e_high
    low_freq_energy_fraction = float(np.sum(psd[freqs < 5.0]) / psd_total)

    return {
        "dominant_freq"           : dominant_freq,
        "central_freq"            : central_freq,
        "energy_band_70"          : energy_band_70,
        "spectral_entropy"        : spectral_entropy,
        "spectral_flatness"       : spectral_flatness,
        "spectral_rolloff"        : spectral_rolloff,
        "peak_count"              : peak_count,
        "spectral_ratio_low_high" : spectral_ratio_low_high,
        "low_freq_energy_fraction": low_freq_energy_fraction,
    }


def features_mfcc(signal: np.ndarray, sr: float) -> dict:
    signal_f32 = signal.astype(np.float32)
    n_fft      = max(32, min(512, len(signal_f32)))
    hop_len    = max(8, n_fft // 4)
    try:
        mfcc_matrix = librosa.feature.mfcc(
            y=signal_f32, sr=int(sr),
            n_mfcc=cfg.N_MFCC, n_fft=n_fft, hop_length=hop_len, center=False,
        )
    except Exception:
        mfcc_matrix = np.zeros((cfg.N_MFCC, 1))

    result = {}
    for i in range(cfg.N_MFCC):
        result[f"mfcc_mean_{i+1}"] = float(np.mean(mfcc_matrix[i]))
        result[f"mfcc_std_{i+1}"]  = float(np.std(mfcc_matrix[i]))
    return result


def features_additional(signal: np.ndarray, sr: float) -> dict:
    n        = len(signal)
    sr_f     = float(sr)
    envelope = np.abs(hilbert(signal))


    noise_end    = max(1, int(n * cfg.NOISE_WINDOW_FRACTION))
    noise_rms    = float(np.sqrt(np.mean(signal[:noise_end] ** 2))) + 1e-12
    signal_rms   = float(np.sqrt(np.mean(signal ** 2)))
    snr_estimate = 20.0 * np.log10(signal_rms / noise_rms)

    envelope_peak_time = float(np.argmax(envelope) / sr_f)

    onset_end   = min(max(cfg.ONSET_MIN_SAMPLES, int(cfg.ONSET_WINDOW_SEC * sr_f)), n)
    env_squared = envelope[:onset_end] ** 2
    t_onset     = np.arange(onset_end) / sr_f
    try:
        onset_slope = float(np.polyfit(t_onset, env_squared, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        onset_slope = float("nan")

    win_len   = max(int(cfg.DOMFREQ_WINDOW_SEC * sr_f), 16)
    hop_size  = max(1, int(win_len * (1.0 - cfg.DOMFREQ_OVERLAP)))
    nperseg_w = min(cfg.WELCH_NPERSEG, win_len)
    dom_freqs = []
    for start in range(0, n - win_len + 1, hop_size):
        seg = signal[start: start + win_len]
        try:
            f_w, p_w = welch(seg, fs=sr_f, nperseg=nperseg_w)
            dom_freqs.append(float(f_w[np.argmax(p_w)]))
        except Exception:
            pass
    dominant_freq_stability = float(np.std(dom_freqs)) if len(dom_freqs) >= 2 else 0.0

    above_90    = np.where(envelope >= 0.90 * float(np.max(envelope)))[0]
    attack_time = float(above_90[0] / sr_f) if len(above_90) > 0 else float(n / sr_f)

    inst_freq       = np.diff(np.unwrap(np.angle(hilbert(signal)))) / (2.0 * np.pi / sr_f)
    inst_freq_clean = inst_freq[(inst_freq > 0) & (inst_freq < sr_f / 2.0)]
    inst_freq_std   = float(np.std(inst_freq_clean)) if len(inst_freq_clean) > 10 else float("nan")

    return {
        "snr_estimate"           : snr_estimate,
        "envelope_peak_time"     : envelope_peak_time,
        "onset_slope"            : onset_slope,
        "dominant_freq_stability": dominant_freq_stability,
        "attack_time"            : attack_time,
        "inst_freq_std"          : inst_freq_std,
    }


# Feature Extraction
def extract_features_from_trace(trace) -> dict | None:
    signal, sr = preprocess_trace(trace)
    if len(signal) < int(sr * 0.5):
        return None
    feats = {}
    feats.update(features_time_domain(signal, sr))
    feats.update(features_freq_domain(signal, sr))
    feats.update(features_mfcc(signal, sr))
    feats.update(features_additional(signal, sr))

    selected = getattr(cfg, "SELECTED_FEATURES", None)
    if selected is not None:
        feats = {k: v for k, v in feats.items() if k in selected}

    return feats


def _parse_components(param, available):
    if isinstance(param, list):
        return [c.strip().upper() for c in param if c.strip()]
    if isinstance(param, str):
        p = param.strip()
        if p.lower() == "auto": return available
        if "," in p:            return [c.strip().upper() for c in p.split(",") if c.strip()]
        if len(p) > 1:          return [c.upper() for c in p]
        return [p.upper()]
    return available


def _allowed_networks():
    net = getattr(cfg, "NETWORKS", "ALL")
    if isinstance(net, str) and net.upper() == "ALL": return None
    if isinstance(net, str):  return {net.strip().upper()}
    if isinstance(net, list): return {n.strip().upper() for n in net}
    return None


def process_event_dir(event_dir: str) -> list:
    extensions       = (".mseed", ".miniseed", ".ms")
    allowed_nets     = _allowed_networks()
    stations_param   = getattr(cfg, "STATIONS", "ALL")
    components_param = cfg.COMPONENTS
    results          = []

    mseed_files = [
        os.path.join(event_dir, f)
        for f in os.listdir(event_dir)
        if f.lower().endswith(extensions)
    ]

    for filepath in mseed_files:
        try:
            stream = mseed_read(filepath)
        except Exception:
            continue
        if not stream:
            continue

        station_code = stream[0].stats.station.upper()
        network_code = stream[0].stats.network.upper()

        if (isinstance(stations_param, list) and
                station_code not in [s.upper() for s in stations_param]):
            stream.clear(); del stream; continue

        if allowed_nets is not None and network_code not in allowed_nets:
            stream.clear(); del stream; continue

        available_comps = list({tr.stats.channel[-1].upper() for tr in stream})
        requested_comps = _parse_components(components_param, available_comps)

        all_feats = {"station_code": station_code,
                     "file_name"   : os.path.basename(filepath)}
        valid = False

        for comp in requested_comps:
            prefix   = f"comp_{comp}_"
            matching = [tr for tr in stream if tr.stats.channel[-1].upper() == comp]
            if not matching:
                all_feats[f"{prefix}MISSING"] = True
                continue
            feats = extract_features_from_trace(matching[0])
            if feats is None:
                all_feats[f"{prefix}MISSING"] = True
                continue
            valid = True
            for k, v in feats.items():
                all_feats[f"{prefix}{k}"] = v

        stream.clear(); del stream
        if valid:
            results.append(all_feats)

    return results


def _collect_schema(class_event_map: dict) -> list:
    key_set = set()
    for events in class_event_map.values():
        for ev_dir in events[:5]:
            for feats in process_event_dir(ev_dir):
                for k in feats:
                    if not str(k).endswith("MISSING") and k not in ("station_code", "file_name"):
                        key_set.add(k)
    return sorted(key_set)


def build_dataframe(class_event_map: dict,
                    output_path: str | None = None,
                    desc: str = "Extracting") -> pd.DataFrame:
    all_tasks = [
        (label, ev_dir)
        for label, events in sorted(class_event_map.items())
        for ev_dir in events
    ]
    if not all_tasks:
        return pd.DataFrame()

    schema = _collect_schema(class_event_map)
    if not schema:
        print("  [ERROR] Could not determine feature schema.")
        return pd.DataFrame()

    records = []
    for label, event_dir in tqdm(all_tasks, desc=f"  {desc}", unit="event"):
        for feats in process_event_dir(event_dir):
            row = {}
            for k, v in feats.items():
                if str(k).endswith("MISSING"):
                    prefix = k.replace("MISSING", "")
                    for ref in schema:
                        if ref.startswith(prefix):
                            row[ref] = np.nan
                else:
                    row[k] = v
            for ref in schema:
                if ref not in row:
                    row[ref] = np.nan

            row["label"]       = label
            row["event_id"]    = os.path.basename(event_dir)
            row["event_group"] = base_event_id(row["event_id"])
            records.append(row)

    if not records:
        return pd.DataFrame()

    meta = ["label", "event_id", "event_group", "station_code", "file_name"]
    df   = pd.DataFrame(records)
    df   = df[[c for c in meta if c in df.columns] +
              [c for c in df.columns if c not in meta]].reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)

    return df


# ==============================================================================
# DATA PARTITIONING
# ==============================================================================
def split_and_save_folds(df_all: pd.DataFrame) -> dict:
    all_stats = {}
    for partition_idx in range(cfg.N_PARTITIONS):
        partition_num = partition_idx + 1
        seed          = cfg.PARTITION_SEEDS[partition_idx]
        partition_dir = os.path.join(cfg.OUTPUT_PARTITIONS, f"Partition_{partition_num}")
        ctx_path      = os.path.join(partition_dir, "context_80", "features_context.parquet")
        test_dir      = os.path.join(partition_dir, "test_20")

        os.makedirs(os.path.dirname(ctx_path), exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        print(f"\n  {'='*58}")
        print(f"  PARTITION {partition_num} / {cfg.N_PARTITIONS}  (seed={seed})")
        print(f"  {'='*58}")

        groups = (
            df_all[["event_group", "label"]]
            .drop_duplicates(subset="event_group")
            .reset_index(drop=True)
        )

        df_grp_ctx, df_grp_tst = train_test_split(
            groups,
            train_size   = cfg.PARTITION_CTX_FRAC,
            random_state = seed,
            shuffle      = True,
            stratify     = groups["label"],
        )

        train_groups = set(df_grp_ctx["event_group"])

        df_ctx = df_all[df_all["event_group"].isin(train_groups)].copy()
        df_tst = df_all[~df_all["event_group"].isin(train_groups)].copy()

        df_ctx.to_parquet(ctx_path, index=False)
        print(f"\n  Context Partition {partition_num}:")
        print(f"    Rows          : {len(df_ctx):,}")
        print(f"    Unique groups : {df_ctx['event_group'].nunique():,}")
        print(f"    {'Class':<8}  {'Rows':>8}  {'Groups':>8}")
        print(f"    {'-'*8}  {'-'*8}  {'-'*8}")
        for cls in sorted(df_ctx["label"].unique()):
            mask  = df_ctx["label"] == cls
            cnt   = int(mask.sum())
            grps  = int(df_ctx.loc[mask, "event_group"].nunique())
            print(f"    {cls:<8}  {cnt:>8,}  {grps:>8,}")
        print(f"    {'-'*8}  {'-'*8}  {'-'*8}")
        print(f"    {'TOTAL':<8}  {len(df_ctx):>8,}  {df_ctx['event_group'].nunique():>8,}")

        print(f"\n  Test Partition {partition_num}:")
        print(f"    {'Class':<8}  {'Rows':>8}  {'Groups':>8}  {'File'}")
        print(f"    {'-'*8}  {'-'*8}  {'-'*8}  {'-'*30}")
        total_tst = 0
        for cls in sorted(df_tst["label"].unique()):
            df_cls = df_tst[df_tst["label"] == cls]
            out    = os.path.join(test_dir, f"{cls}.parquet")
            df_cls.to_parquet(out, index=False)
            grps   = int(df_cls["event_group"].nunique())
            print(f"    {cls:<8}  {len(df_cls):>8,}  {grps:>8,}  {cls}.parquet")
            total_tst += len(df_cls)
        print(f"    {'-'*8}  {'-'*8}  {'-'*8}")
        print(f"    {'TOTAL':<8}  {total_tst:>8,}  {df_tst['event_group'].nunique():>8,}")

        stats_key = f"fold{partition_num}"
        all_stats[f"{stats_key}_ctx_rows"]    = len(df_ctx)
        all_stats[f"{stats_key}_tst_rows"]    = total_tst
        all_stats[f"{stats_key}_seed"]        = seed
        all_stats[f"{stats_key}_ctx_groups"]  = int(df_ctx["event_group"].nunique())
        all_stats[f"{stats_key}_tst_groups"]  = int(df_tst["event_group"].nunique())
        for cls in sorted(df_ctx["label"].unique()):
            all_stats[f"{stats_key}_ctx_{cls}"] = int((df_ctx["label"] == cls).sum())
            all_stats[f"{stats_key}_tst_{cls}"] = int((df_tst["label"] == cls).sum())

    return all_stats


def print_final_summary(class_events: dict, elapsed: float) -> None:
    print(f"\n  {'='*58}")
    print(f"  FINAL SUMMARY")
    print(f"  {'='*58}")
    total_test_global = 0
    for partition_num in range(1, cfg.N_PARTITIONS + 1):
        partition_dir = os.path.join(cfg.OUTPUT_PARTITIONS, f"Partition_{partition_num}")
        ctx_path = os.path.join(partition_dir, "context_80", "features_context.parquet")
        test_dir = os.path.join(partition_dir, "test_20")
        if not os.path.exists(ctx_path):
            print(f"\n  Partition {partition_num}: [NOT GENERATED]")
            continue
        df_ctx    = pd.read_parquet(ctx_path)
        total_tst = 0
        test_info = {}
        if os.path.isdir(test_dir):
            for tf in sorted(os.listdir(test_dir)):
                if tf.endswith(".parquet"):
                    n = len(pd.read_parquet(os.path.join(test_dir, tf)))
                    test_info[tf.replace(".parquet", "")] = n
                    total_tst += n
        total_test_global += total_tst
        print(f"\n  Partition {partition_num}:")
        print(f"    {'Class':<8}  {'Context':>10}  {'Test':>8}")
        print(f"    {'-'*8}  {'-'*10}  {'-'*8}")
        for cls in sorted(set(list(df_ctx["label"].unique()) + list(test_info.keys()))):
            ctx_n = int(df_ctx["label"].value_counts().get(cls, 0))
            tst_n = test_info.get(cls, 0)
            print(f"    {cls:<8}  {ctx_n:>10,}  {tst_n:>8,}")
        print(f"    {'-'*8}  {'-'*10}  {'-'*8}")
        print(f"    {'TOTAL':<8}  {len(df_ctx):>10,}  {total_tst:>8,}")
    print(f"\n  Total test rows (all partitions): {total_test_global:,}")
    print(f"  Total time: {elapsed / 3600:.2f} hours")
    print(f"\n  {'='*58}")
    print(f"  PIPELINE COMPLETED")
    print(f"  {'='*58}\n")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main() -> None:
    np.random.seed(cfg.PARTITION_SEEDS[0])
    start_time = time.time()
    W = 58
    print(f"\n  {'='*W}")
    print(f"  FEATURE EXTRACTION - GALERAS")
    print(f"  {'='*W}")
    print(f"  Input  : {cfg.INPUT_ROOT}")
    print(f"  Output : {cfg.OUTPUT_PARTITIONS}")
    print(f"  {'='*W}\n")

    print("[STEP 1] Discovering miniSEED events...")
    class_events = discover_events(cfg.INPUT_ROOT)
    if not class_events:
        print(f"[ERROR] No events found in: {cfg.INPUT_ROOT}")
        return
    total_eventos = sum(len(v) for v in class_events.values())
    print(f"\n  {'Class':<8}  {'Events':>10}")
    print(f"  {'-'*8}  {'-'*10}")
    for cls, evs in sorted(class_events.items()):
        print(f"  {cls:<8}  {len(evs):>10,}")
    print(f"  {'-'*8}  {'-'*10}")
    print(f"  {'TOTAL':<8}  {total_eventos:>10,}")

    epc    = getattr(cfg, 'EVENTS_PER_CLASS', None)
    max_ev = getattr(cfg, 'MAX_TRZ', None)
    if epc is not None or max_ev is not None:
        msg = f"{epc:,} per class" if epc else f"limit of {max_ev:,}"
        print(f"\n[STEP 2] Sampling events ({msg})...")
        class_events = sample_events(class_events)
        total_sam = sum(len(v) for v in class_events.values())
        print(f"\n  {'Class':<8}  {'Events':>10}")
        print(f"  {'-'*8}  {'-'*10}")
        for cls, evs in sorted(class_events.items()):
            print(f"  {cls:<8}  {len(evs):>10,}")
        print(f"  {'-'*8}  {'-'*10}")
        print(f"  {'TOTAL':<8}  {total_sam:>10,}")
    else:
        print(f"\n[STEP 2] Using all events (no sampling).")

    print(f"\n[STEP 3] Extracting features from {sum(len(v) for v in class_events.values()):,} events...")
    df_all = build_dataframe(class_events, desc="Extracting features")
    if df_all.empty:
        print("[ERROR] No features extracted.")
        return
    print(f"\n  Extracted features: {len(df_all):,} rows")
    print(f"  Unique groups (event_group): {df_all['event_group'].nunique():,}")
    print(f"  {'Class':<8}  {'Rows':>8}  {'Groups':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}")
    for cls in sorted(df_all["label"].unique()):
        mask = df_all["label"] == cls
        cnt  = int(mask.sum())
        grps = int(df_all.loc[mask, "event_group"].nunique())
        print(f"  {cls:<8}  {cnt:>8,}  {grps:>8,}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'TOTAL':<8}  {len(df_all):>8,}  {df_all['event_group'].nunique():>8,}")

    print(f"\n[STEP 4] Splitting into {cfg.N_PARTITIONS} partitions and saving parquets...")
    all_stats = split_and_save_folds(df_all)
    elapsed = time.time() - start_time
    print_final_summary(class_events, elapsed)


if __name__ == "__main__":
    main()