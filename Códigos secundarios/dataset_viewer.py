import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from obspy import read as mseed_read

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input dataset directory
SISMOS_DIR   = # Dataset access path

# Output directory for generated images
OUTPUT_DIR   = r"C:\Users\Daniel\OneDrive\Documentos\Tesis_Galeras_Final\eda_imagenes"
MD_OUTPUT    = os.path.join(OUTPUT_DIR, "metadata_eda.md")

STATION      = "ANGP"
NETWORK      = "CM"
CHANNEL      = "ELZ"

CLASSES      = ["LP", "TRE", "VT", "TOR"]
CLASS_NAMES  = {
    "LP": "Long Period (LP)",
    "TRE": "Tremor (TRE)",
    "VT": "Volcano-Tectonic (VT)",
    "TOR": "Tornillo (TO)"
}

FIG_DPI      = 150
FIG_WIDTH    = 10
FIG_HEIGHT   = 8

COLOR_SIGNAL = "#072F5F"
COLOR_PSD    = "#072F5F"

# ==============================================================================
# RANDOM EVENT SEARCH
# ==============================================================================

def _match_file(filename: str) -> bool:
    if not filename.lower().endswith((".mseed", ".miniseed", ".ms")):
        return False
    up = filename.upper()
    return STATION in up and CHANNEL in up

def find_random_event(event_class: str) -> tuple:
    class_path = os.path.join(SISMOS_DIR, event_class)
    if not os.path.exists(class_path):
        return None
    years = [y for y in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, y))]
    if not years:
        return None
    for _ in range(200):
        yr = random.choice(years)
        yr_path = os.path.join(class_path, yr)
        events = [e for e in os.listdir(yr_path) if os.path.isdir(os.path.join(yr_path, e))]
        if not events: continue
        evt = random.choice(events)
        evt_path = os.path.join(yr_path, evt)
        files = [f for f in os.listdir(evt_path) if _match_file(f)]
        if files:
            fname = random.choice(files)
            return (os.path.join(evt_path, fname), yr, evt, fname)
    return None

# ==============================================================================
# MULTIPANEL PLOT GENERATION
# ==============================================================================

def plot_eda(filepath: str, event_class: str, out_path: str):
    tr = mseed_read(filepath)[0]
    tr.detrend("constant")
    tr.detrend("linear")
    if abs(tr.stats.sampling_rate - 100.0) > 0.1:
        tr.resample(100.0)
    tr.filter("highpass", freq=0.7, zerophase=True)
    data = tr.data.astype(np.float64)
    sr   = tr.stats.sampling_rate
    peak_idx = int(np.argmax(np.abs(data)))
    samples_before = int(10 * sr) 
    samples_after = int(30 * sr)
    start_idx = max(0, peak_idx - samples_before)
    end_idx = min(len(data), peak_idx + samples_after)
    data = data[start_idx:end_idx]
    t = np.arange(len(data)) / sr
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.subplots_adjust(hspace=0.4)
    class_title = CLASS_NAMES.get(event_class, event_class.upper())
    fig.suptitle(f"Class: {class_title} | Station: {STATION} | Channel: {CHANNEL}", fontsize=14, fontweight="bold")
    ax1.plot(t, data, color=COLOR_SIGNAL, linewidth=1.2)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_xlabel("Time [s]", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.margins(x=0)
    nperseg = min(1024, len(data))
    freqs, psd = welch(data, fs=sr, nperseg=nperseg)
    ax2.plot(freqs, psd, color=COLOR_PSD, linewidth=1.2)
    ax2.set_ylabel("PSD", fontsize=10)
    ax2.set_xlabel("Frequency [Hz]", fontsize=10)
    ax2.set_xlim([0, 25])
    ax2.set_ylim(bottom=0)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.grid(True, alpha=0.3)
    ax2.margins(x=0)
    NFFT = min(256, len(data)//2)
    ax3.specgram(data, NFFT=NFFT, Fs=sr, noverlap=NFFT//2, cmap="viridis")
    ax3.set_ylabel("Frequency [Hz]", fontsize=10)
    ax3.set_ylim([0, 25])
    ax3.set_xlabel("Time [s]", fontsize=10)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("Starting Exploratory Data Analysis (EDA) generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Number of sets to generate
    NUM_RUNS = 10
    for run in range(1, NUM_RUNS + 1):
        run_dir = os.path.join(OUTPUT_DIR, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)
        md_output = os.path.join(run_dir, "metadata_eda.md")
        print(f"\n--- GENERATING RUN {run}/{NUM_RUNS} ---")
        with open(md_output, "w", encoding="utf-8") as md_file:
            md_file.write(f"# Signal Metadata for Visual Inspection (Run {run})\n\n")
            for event_class in CLASSES:
                print(f"Searching for representative event for class: {event_class.upper()}...")
                candidate = find_random_event(event_class)
                if candidate is None:
                    print(f"  [ERROR] No valid signal found for {event_class}.")
                    continue
                fp, yr, evt, fname = candidate
                out_img = os.path.join(run_dir, f"eda_{event_class}.png")
                print(f"  -> Selected: {yr}/{evt}/{fname}")
                try:
                    plot_eda(fp, event_class, out_img)
                    class_name = CLASS_NAMES.get(event_class, event_class.upper())
                    md_file.write(f"## Class {class_name}\n")
                    md_file.write(f"- **Generated image:** `eda_{event_class}.png`\n")
                    md_file.write(f"- **Year folder:** `{yr}`\n")
                    md_file.write(f"- **Event subfolder:** `{evt}`\n")
                    md_file.write(f"- **.mseed file:** `{fname}`\n")
                    md_file.write(f"- **Absolute path:** `{fp}`\n")
                    md_file.write(f"- **Station:** `{STATION}` | **Channel:** `{CHANNEL}`\n\n")
                    print("  [OK] Plot saved.")
                except Exception as e:
                    print(f"  [ERROR] Generation failed for {event_class}: {e}")
    print(f"\nProcess finished. Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
