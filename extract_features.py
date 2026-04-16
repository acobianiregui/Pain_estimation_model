"""
Biosignal Feature Extraction for Pain Classification
=====================================================
Extracts features from sliding/non-overlapping windows across subjects 1–52.
Input : One CSV per subject (comma-delimited), sampled at 250 Hz (0.004 s).
Output: One feature CSV per subject + one combined CSV in the output folder.

Columns expected: Seconds,Ecg,Eda_E4,Eda_RB,Bvp,Emg,Resp,COVAS,Tmp

Entry point:
    get_features(input_folder, output_folder, window_length,
                 overlap=False, overlap_percentage=0.0)

Dependencies: pip install numpy pandas scipy
"""

import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings("ignore")

import re

FS = 250  # sampling frequency (Hz) — fixed by the hardware

# ── Filter helpers ────────────────────────────────────────────────────────────

def bandpass(data, low, high, fs=FS, order=4):
    """Used to isolate the breathing band in respiration (0.1–1.0 Hz)."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

def lowpass(data, cutoff, fs=FS, order=4):
    """Used for EDA decomposition only — separates tonic SCL from phasic SCR."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)

# ── Statistical helpers ───────────────────────────────────────────────────────

def safe_entropy(x, m=2, r_factor=0.2):
    """Approximate sample entropy (fast scalar implementation)."""
    x = np.array(x, dtype=float)
    r = r_factor * np.std(x, ddof=1)
    if r == 0:
        return 0.0
    n = len(x)
    def phi(m_len):
        count = 0
        templates = np.array([x[i:i + m_len] for i in range(n - m_len + 1)])
        for i in range(len(templates)):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dists <= r) - 1  # exclude self
        return count / (n - m_len + 1)
    phi_m  = phi(m)
    phi_m1 = phi(m + 1)
    if phi_m == 0:
        return 0.0
    return -np.log(phi_m1 / phi_m) if phi_m1 > 0 else 0.0

def basic_stats(arr, prefix):
    """Return dict of mean, std, min, max, skew, kurtosis for an array."""
    return {
        f"{prefix}_mean": np.mean(arr),
        f"{prefix}_std" : np.std(arr, ddof=1),
        f"{prefix}_min" : np.min(arr),
        f"{prefix}_max" : np.max(arr),
        f"{prefix}_skew": float(stats.skew(arr)),
        f"{prefix}_kurt": float(stats.kurtosis(arr)),
    }

# ── Per-signal feature extractors ────────────────────────────────────────────

def ecg_features(ecg_window, fs=FS):
    feats = {}
    feats.update(basic_stats(ecg_window, "ecg"))

    # Initialise all derived features to NaN
    for k in ["ecg_hr_bpm","ecg_rr_mean_ms","ecg_rr_std_ms","ecg_rmssd",
              "ecg_rr_range_ms","ecg_pnn50"]:
        feats[k] = np.nan

    # R-peak detection
    try:
        height_thresh = np.mean(ecg_window) + 0.5 * np.std(ecg_window)
        peaks, _ = find_peaks(ecg_window,
                              distance=int(0.3 * fs),
                              height=height_thresh)

        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / fs * 1000  # ms
            feats["ecg_hr_bpm"]      = 60000 / np.mean(rr_intervals)
            feats["ecg_rr_mean_ms"]  = np.mean(rr_intervals)
            feats["ecg_rr_std_ms"]   = np.std(rr_intervals, ddof=1)
            feats["ecg_rmssd"]       = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            feats["ecg_rr_range_ms"] = np.ptp(rr_intervals)
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            feats["ecg_pnn50"]       = nn50 / len(rr_intervals) * 100
    except Exception:
        pass

    return feats


def eda_features(eda_window, prefix, fs=FS):
    """Shared EDA extractor — works for both E4 and RB placement."""
    feats = {}
    feats.update(basic_stats(eda_window, prefix))

    for k in [f"{prefix}_scl_mean", f"{prefix}_scl_std", f"{prefix}_scl_slope",
              f"{prefix}_scr_n_peaks", f"{prefix}_scr_mean_amp", f"{prefix}_scr_max_amp"]:
        feats[k] = np.nan

    try:
        # Decompose: SCL = low-pass <0.05 Hz; SCR = residual
        scl = lowpass(eda_window, 0.05, fs)
        scr = eda_window - scl

        # SCL features
        feats[f"{prefix}_scl_mean"]  = np.mean(scl)
        feats[f"{prefix}_scl_std"]   = np.std(scl, ddof=1)
        slope = np.polyfit(np.arange(len(scl)), scl, 1)[0]
        feats[f"{prefix}_scl_slope"] = slope

        # SCR features — detect peaks in phasic component
        scr_rect = np.maximum(scr, 0)
        peaks, props = find_peaks(scr_rect,
                                  height=0.01 * np.ptp(eda_window) if np.ptp(eda_window) > 0 else 0.001,
                                  distance=int(0.5 * fs))
        feats[f"{prefix}_scr_n_peaks"] = len(peaks)
        if len(peaks) > 0:
            feats[f"{prefix}_scr_mean_amp"] = np.mean(props["peak_heights"])
            feats[f"{prefix}_scr_max_amp"]  = np.max(props["peak_heights"])
        else:
            feats[f"{prefix}_scr_mean_amp"] = 0.0
            feats[f"{prefix}_scr_max_amp"]  = 0.0

    except Exception:
        pass

    return feats


def bvp_features(bvp_window, fs=FS):
    feats = {}
    feats.update(basic_stats(bvp_window, "bvp"))

    for k in ["bvp_ac_dc_ratio","bvp_hr_bpm","bvp_ibi_mean_ms","bvp_ibi_std_ms",
              "bvp_ibi_cv","bvp_pulse_amp_mean","bvp_pulse_amp_std","bvp_vasomotor_power"]:
        feats[k] = np.nan

    try:
        peaks, _ = find_peaks(bvp_window,
                              distance=int(0.4 * fs),
                              height=np.mean(bvp_window))

        # AC/DC ratio
        ac = np.ptp(bvp_window)
        dc = np.mean(np.abs(bvp_window))
        feats["bvp_ac_dc_ratio"] = ac / dc if dc > 0 else np.nan

        if len(peaks) >= 2:
            ibi = np.diff(peaks) / fs * 1000  # ms
            feats["bvp_hr_bpm"]         = 60000 / np.mean(ibi)
            feats["bvp_ibi_mean_ms"]    = np.mean(ibi)
            feats["bvp_ibi_std_ms"]     = np.std(ibi, ddof=1)
            feats["bvp_ibi_cv"]         = feats["bvp_ibi_std_ms"] / feats["bvp_ibi_mean_ms"] if feats["bvp_ibi_mean_ms"] > 0 else np.nan
            feats["bvp_pulse_amp_mean"] = np.mean(bvp_window[peaks])
            feats["bvp_pulse_amp_std"]  = np.std(bvp_window[peaks], ddof=1)

        # Frequency domain: vasomotor band
        freqs, psd = signal.welch(bvp_window, fs=fs, nperseg=min(len(bvp_window), 512))
        vm_mask = (freqs >= 0.05) & (freqs < 0.15)
        feats["bvp_vasomotor_power"] = np.trapz(psd[vm_mask], freqs[vm_mask]) if vm_mask.any() else 0.0

    except Exception:
        pass

    return feats


def emg_features(emg_window, fs=FS):
    feats = {}
    feats.update(basic_stats(emg_window, "emg"))

    for k in ["emg_rms","emg_mav","emg_iemg","emg_wl","emg_zcr",
              "emg_median_freq","emg_mean_freq"]:
        feats[k] = np.nan

    try:
        emg_rect = np.abs(emg_window)

        # Time-domain amplitude
        feats["emg_rms"]   = np.sqrt(np.mean(emg_window ** 2))
        feats["emg_mav"]   = np.mean(emg_rect)
        feats["emg_iemg"]  = np.sum(emg_rect) / fs
        feats["emg_wl"]    = np.sum(np.abs(np.diff(emg_window)))  # waveform length

        # Zero crossing rate
        zc = np.sum(np.diff(np.sign(emg_window)) != 0)
        feats["emg_zcr"] = zc / len(emg_window)

        # Frequency domain
        freqs, psd = signal.welch(emg_window, fs=fs, nperseg=min(len(emg_window), 512))
        pos = freqs > 0
        if pos.any():
            cum_pow   = np.cumsum(psd[pos]) / np.sum(psd[pos])
            feats["emg_median_freq"] = freqs[pos][np.searchsorted(cum_pow, 0.5)]
            feats["emg_mean_freq"]   = np.sum(freqs[pos] * psd[pos]) / np.sum(psd[pos]) if np.sum(psd[pos]) > 0 else np.nan

    except Exception:
        pass

    return feats


def resp_features(resp_window, fs=FS):
    feats = {}
    feats.update(basic_stats(resp_window, "resp"))

    for k in ["resp_rate_bpm","resp_interval_mean_s","resp_interval_range_s",
              "resp_irregularity_idx","resp_amp_mean","resp_amp_std","resp_dominant_freq"]:
        feats[k] = np.nan

    try:
        resp_filt = bandpass(resp_window, 0.1, 1.0, fs)

        # Breath peaks (inhalation peaks)
        peaks, _   = find_peaks(resp_filt, distance=int(1.5 * fs))
        troughs, _ = find_peaks(-resp_filt, distance=int(1.5 * fs))

        if len(peaks) >= 2:
            peak_times       = peaks / fs
            breath_intervals = np.diff(peak_times)  # seconds
            feats["resp_rate_bpm"]         = 60 / np.mean(breath_intervals)
            feats["resp_interval_mean_s"]  = np.mean(breath_intervals)
            # Range of breath-to-breath intervals — rises with irregular breathing
            feats["resp_interval_range_s"] = np.ptp(breath_intervals)
            # Coefficient of variation using range — normalised irregularity measure
            feats["resp_irregularity_idx"] = (
                np.ptp(breath_intervals) / np.mean(breath_intervals)
                if np.mean(breath_intervals) > 0 else np.nan
            )

        # Tidal volume proxy
        if len(peaks) > 0 and len(troughs) > 0:
            amp_list = []
            for p in peaks:
                nearby = troughs[np.argmin(np.abs(troughs - p))]
                amp_list.append(abs(resp_filt[p] - resp_filt[nearby]))
            feats["resp_amp_mean"] = np.mean(amp_list)
            feats["resp_amp_std"]  = np.std(amp_list, ddof=1) if len(amp_list) > 1 else 0.0
        else:
            feats["resp_amp_mean"] = np.ptp(resp_filt)
            feats["resp_amp_std"]  = 0.0

        # Dominant frequency
        freqs, psd = signal.welch(resp_filt, fs=fs, nperseg=min(len(resp_filt), 512))
        resp_band  = (freqs >= 0.1) & (freqs <= 1.0)
        if resp_band.any():
            feats["resp_dominant_freq"] = freqs[resp_band][np.argmax(psd[resp_band])]

    except Exception:
        pass

    return feats


# ── Window-level feature extraction ──────────────────────────────────────────

def extract_window_features(df_window, window_idx):
    row = {"window_idx": window_idx}

    t_start = df_window["Seconds"].iloc[0]
    t_end   = df_window["Seconds"].iloc[-1]
    row["t_start_s"] = t_start
    row["t_end_s"]   = t_end

    # COVAS summary statistics for the window
    covas_vals = df_window["COVAS"].values
    row["covas_mean"] = np.nanmean(covas_vals)
    row["covas_max"]  = np.nanmax(covas_vals)
    row["covas_min"]  = np.nanmin(covas_vals)
    row["covas_diff"] = row["covas_max"] - row["covas_min"]

    # Covas Delta statistics
    midp = len(covas_vals) // 2
    firstHalf = covas_vals[:midp]
    secondHalf = covas_vals[midp:]
    firstAvg = np.nanmean(firstHalf)
    secondAvg = np.nanmean(secondHalf)
    row["covas_delta"] = secondAvg-firstAvg
    

    # Per-signal features
    row.update(ecg_features(df_window["Ecg"].values))
    row.update(eda_features(df_window["Eda_E4"].values, "eda_e4"))
    row.update(eda_features(df_window["Eda_RB"].values, "eda_rb"))
    row.update(bvp_features(df_window["Bvp"].values))
    row.update(emg_features(df_window["Emg"].values))
    row.update(resp_features(df_window["Resp"].values))

    # Cross-signal: EDA E4 vs RB correlation
    try:
        e4  = df_window["Eda_E4"].values
        rb  = df_window["Eda_RB"].values
        if np.std(e4) > 0 and np.std(rb) > 0:
            row["cross_eda_e4_rb_corr"] = float(np.corrcoef(e4, rb)[0, 1])
        else:
            row["cross_eda_e4_rb_corr"] = np.nan
    except Exception:
        row["cross_eda_e4_rb_corr"] = np.nan

    # Cross-signal: BVP–ECG HR agreement
    try:
        bvp_hr  = row.get("bvp_hr_bpm", np.nan)
        ecg_hr  = row.get("ecg_hr_bpm", np.nan)
        if not np.isnan(bvp_hr) and not np.isnan(ecg_hr):
            row["cross_hr_bvp_ecg_diff"] = abs(bvp_hr - ecg_hr)
        else:
            row["cross_hr_bvp_ecg_diff"] = np.nan
    except Exception:
        row["cross_hr_bvp_ecg_diff"] = np.nan

    return row


# ── Per-subject processing ────────────────────────────────────────────────────

def process_subject(filepath, subject_id, window_samples, step_samples):
    """Extract windowed features for a single subject CSV.

    Parameters
    ----------
    filepath       : path to the subject's CSV file
    subject_id     : integer subject index (used as subject_idx column value)
    window_samples : number of samples per window
    step_samples   : number of samples to advance between window starts;
                     equals window_samples when there is no overlap
    """
    print(f"  Processing subject {subject_id}...")

    try:
        df = pd.read_csv(filepath, sep=",", dtype=float, low_memory=False)
    except Exception as e:
        print(f"    ERROR reading {filepath}: {e}")
        return None

    # Rename columns defensively (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    required = ["Seconds", "Ecg", "Eda_E4", "Eda_RB", "Bvp", "Emg", "Resp"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"    WARNING: missing columns {missing} in subject {subject_id}. Skipping.")
        return None

    # Drop NaN in signal columns
    sig_cols = ["Ecg", "Eda_E4", "Eda_RB", "Bvp", "Emg", "Resp"]
    df = df.dropna(subset=sig_cols).reset_index(drop=True)

    n_samples = len(df)

    if n_samples < window_samples:
        print(f"    WARNING: subject {subject_id} has fewer than {window_samples} samples. Skipping.")
        return None

    # Build list of window start indices
    starts = list(range(0, n_samples - window_samples + 1, step_samples))

    rows = []
    for w, start in enumerate(starts):
        end       = start + window_samples
        window_df = df.iloc[start:end]
        feats     = extract_window_features(window_df, w)
        rows.append(feats)

    result_df = pd.DataFrame(rows)

    # Add subject index as the first column
    result_df.insert(0, "subject_idx", subject_id)

    # Reorder: subject metadata first, then window metadata, then features
    meta_cols = ["subject_idx", "window_idx", "t_start_s", "t_end_s",
                 "covas_mean", "covas_max", "covas_min", "covas_diff"]
    feat_cols = [c for c in result_df.columns if c not in meta_cols]
    result_df = result_df[meta_cols + feat_cols]

    return result_df


# ── Public API ────────────────────────────────────────────────────────────────

def get_features(
    input_folder: str,
    output_folder: str,
    window_length: float,
    overlap: bool = False,
    overlap_percentage: float = 0.0,
) -> pd.DataFrame:
    """Extract biosignal features for all subjects and save results to CSV files.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing per-subject CSV files.
    output_folder : str
        Path to the folder where per-subject feature CSVs and the combined CSV
        will be written (created automatically if it does not exist).
    window_length : float
        Length of each analysis window in seconds (e.g. 10.0).
    overlap : bool, optional
        Whether consecutive windows should overlap.  Default is False
        (non-overlapping / tumbling windows).
    overlap_percentage : float, optional
        Percentage of a window that is covered by the *next* window when
        ``overlap=True``.  Must be in the range [0, 50).  A value of 50 would
        mean each window starts at the midpoint of the previous one.
        Ignored when ``overlap=False``.  Default is 0.0.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all subjects and all windows.
        Also written to ``<output_folder>/all_subjects_features.csv``.

    Raises
    ------
    ValueError
        If ``window_length`` is not positive, or if ``overlap_percentage`` is
        outside [0, 50).
    FileNotFoundError
        If ``input_folder`` does not exist.
    """

    # ── Parameter validation ──────────────────────────────────────────────────
    if not isinstance(input_folder, str):
        raise TypeError("input_folder must be a string.")
    if not isinstance(output_folder, str):
        raise TypeError("output_folder must be a string.")
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: '{input_folder}'")
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}.")
    if overlap and not (0.0 <= overlap_percentage < 50.0):
        raise ValueError(
            f"overlap_percentage must be in [0, 50), got {overlap_percentage}."
        )

    # ── Derive sample counts ──────────────────────────────────────────────────
    window_samples = int(round(window_length * FS))

    if overlap:
        # step = window minus the overlapping tail
        overlap_samples = int(round(overlap_percentage / 100.0 * window_samples))
        step_samples    = window_samples - overlap_samples
    else:
        step_samples = window_samples   # tumbling (non-overlapping) windows

    os.makedirs(output_folder, exist_ok=True)

    print(f"Window : {window_length} s  ({window_samples} samples)")
    if overlap:
        print(f"Overlap: {overlap_percentage}%  (step = {step_samples} samples)")
    else:
        print("Overlap: none")

    # ── Discover subject files ────────────────────────────────────────────────
    csv_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".csv")
    ])

    if not csv_files:
        print(f"No CSV files found in '{input_folder}'. Check input_folder path.")
        return pd.DataFrame()

    print(f"\nFound {len(csv_files)} CSV file(s) in '{input_folder}'.\n")

    # ── Process each subject ──────────────────────────────────────────────────
    processed    = 0
    all_subjects = []

    for fname in csv_files:
        nums       = re.findall(r"\d+", fname)
        subject_id = int(nums[0]) if nums else fname

        filepath = os.path.join(input_folder, fname)
        result   = process_subject(filepath, subject_id, window_samples, step_samples)

        if result is not None:
            out_path = os.path.join(output_folder, f"subject_{subject_id:02d}_features.csv")
            result.to_csv(out_path, index=False)
            print(f"    Saved {len(result)} windows → {out_path}")
            all_subjects.append(result)
            processed += 1

    # ── Combine and write master CSV ──────────────────────────────────────────
    if not all_subjects:
        print("\nNo subjects were processed successfully.")
        return pd.DataFrame()

    combined = pd.concat(all_subjects, ignore_index=True)
    combined.sort_values(["subject_idx", "window_idx"], inplace=True)
    combined_path = os.path.join(output_folder, "all_subjects_features.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined CSV ({len(combined)} rows, {len(combined.columns)} columns) → {combined_path}")
    print(f"\nDone. {processed}/{len(csv_files)} subjects processed.")
    print(f"Feature CSVs written to '{output_folder}/'.")

    return combined


# ── Backwards-compatible entry point ─────────────────────────────────────────

def main():
    """Run with the original default settings (10 s, no overlap)."""
    get_features(
        input_folder="./data",
        output_folder="./features",
        window_length=10.0,
        overlap=False,
    )


if __name__ == "__main__":
    main()
