import os

# ======================= CONFIGURATION =======================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_ROOT = # Input dataset directory path

# Output directory path for processed partitions
OUTPUT_PARTITIONS = os.path.join(_BASE_DIR, "data_processed", "partitions")

# --- Dataset Filters ---
COMPONENTS = "Z"
STATIONS = ["ANGP"]
NETWORKS = "CM"

# --- Partitioning Configuration ---
N_PARTITIONS = 4
PARTITION_CTX_FRAC = 0.80
MAX_TRZ = 12500
PARTITION_SEEDS = [42, 123, 456, 789]

# --- Preprocessing Filters ---
TARGET_SAMPLERATE = 100.0
HIGHPASS_FREQ = 0.7

# --- Time Domain Feature Configuration ---
NOISE_WINDOW_FRACTION = 0.10
ONSET_WINDOW_SEC = 1.5
ONSET_MIN_SAMPLES = 5
CODA_THRESHOLD = 0.10
SNR_MIN_FOR_CODA = 3.0

# --- Frequency Domain Feature Configuration ---
WELCH_NPERSEG = 256
SPECTRAL_ROLLOFF = 0.85
ENERGY_BAND_PCT = 0.70
DOMFREQ_WINDOW_SEC = 2.0
DOMFREQ_OVERLAP = 0.50
SPECRATIO_LOW_MIN = 1.0
SPECRATIO_LOW_MAX = 3.0
SPECRATIO_HIGH_MIN = 8.0
SPECRATIO_HIGH_MAX = 20.0

# ---Mel cepstral coefficients (MFCC) --
N_MFCC = 13

# ======================= FEATURES =======================
# Note: By setting this to None, all 53 extracted features will be generated.
# SELECTED_FEATURES = None

SELECTED_FEATURES = [
    "duration",
    "spectral_entropy",
    "spectral_ratio_low_high",
    "kurtosis",
    "envelope_skewness",
    "central_freq",
    "mfcc_mean_2",
    "peak_count",
    "energy_ratio_first_half",
    "mfcc_mean_8",
    "mfcc_mean_4",
    "snr_estimate",
    "envelope_peak_time",
    "mfcc_mean_9",
    "spectral_rolloff",
    "low_freq_energy_fraction",
    "dominant_freq",
]
