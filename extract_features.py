"""
Biosignal Feature Extraction for Pain Classification
=====================================================
Extracts features from 10-second non-overlapping windows across subjects 1–52.
Input : One CSV per subject (semicolon-delimited), sampled at 250 Hz (0.004s).
Output: One feature CSV per subject in the specified output folder.

Columns expected: Seconds;Ecg;Eda_E4;Eda_RB;Bvp;Emg;Resp;COVAS;Tmp

Dependencies: pip install numpy pandas scipy neurokit2
"""

import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_FOLDER  = "./ECE4782_project_data/Filtered"          # folder containing subject CSVs
OUTPUT_FOLDER = "./Features"      # folder where feature CSVs will be saved
FS            = 250               # sampling frequency (1 / 0.004 s)
WINDOW_SEC    = 10                # non-overlapping window length in seconds
WINDOW_SAMPLES = FS * WINDOW_SEC  # = 2500 samples per window

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
    # Basic stats on raw signal
    feats.update(basic_stats(ecg_window, "ecg"))

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
            feats["ecg_rr_std_ms"]   = np.std(rr_intervals, ddof=1)   # SDNN
            feats["ecg_rmssd"]       = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            feats["ecg_rr_range_ms"] = np.ptp(rr_intervals)
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            feats["ecg_pnn50"]       = nn50 / len(rr_intervals) * 100

            # Poincaré SD1 / SD2
            d   = np.diff(rr_intervals)
            sd1 = np.std(d / np.sqrt(2), ddof=1)
            sd2 = np.std(rr_intervals - np.mean(rr_intervals) + d / np.sqrt(2), ddof=1)
            feats["ecg_poincare_sd1"] = sd1
            feats["ecg_poincare_sd2"] = sd2

            # HRV frequency domain (Welch on RR series if enough beats)
            if len(rr_intervals) >= 4:
                rr_fs = 4  # resample RR to 4 Hz for spectral analysis
                rr_time = np.cumsum(rr_intervals) / 1000
                rr_time -= rr_time[0]
                t_uniform = np.arange(0, rr_time[-1], 1 / rr_fs)
                rr_interp = np.interp(t_uniform, rr_time, rr_intervals)
                freqs, psd = signal.welch(rr_interp, fs=rr_fs, nperseg=min(len(rr_interp), 64))
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                lf_pow  = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else 0
                hf_pow  = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else 0
                feats["ecg_lf_power"]  = lf_pow
                feats["ecg_hf_power"]  = hf_pow
                feats["ecg_lf_hf_ratio"] = lf_pow / hf_pow if hf_pow > 0 else np.nan
        else:
            # Not enough peaks — fill with NaN
            for k in ["ecg_hr_bpm","ecg_rr_mean_ms","ecg_rr_std_ms","ecg_rmssd",
                      "ecg_rr_range_ms","ecg_pnn50","ecg_poincare_sd1",
                      "ecg_poincare_sd2","ecg_lf_power","ecg_hf_power","ecg_lf_hf_ratio"]:
                feats[k] = np.nan
    except Exception:
        pass

    return feats


def eda_features(eda_window, prefix, fs=FS):
    """Shared EDA extractor — works for both E4 and RB placement."""
    feats = {}
    feats.update(basic_stats(eda_window, prefix))

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
            feats[f"{prefix}_scr_auc"]      = np.trapz(scr_rect)
        else:
            feats[f"{prefix}_scr_mean_amp"] = 0.0
            feats[f"{prefix}_scr_max_amp"]  = 0.0
            feats[f"{prefix}_scr_auc"]      = np.trapz(scr_rect)

        # Signal magnitude area
        feats[f"{prefix}_sma"] = np.sum(np.abs(eda_window)) / len(eda_window)

    except Exception:
        pass

    return feats


def bvp_features(bvp_window, fs=FS):
    feats = {}
    feats.update(basic_stats(bvp_window, "bvp"))

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
            feats["bvp_hr_bpm"]      = 60000 / np.mean(ibi)
            feats["bvp_ibi_mean_ms"] = np.mean(ibi)
            feats["bvp_ibi_std_ms"]  = np.std(ibi, ddof=1)
            feats["bvp_ibi_cv"]      = feats["bvp_ibi_std_ms"] / feats["bvp_ibi_mean_ms"] if feats["bvp_ibi_mean_ms"] > 0 else np.nan
            feats["bvp_pulse_amp_mean"] = np.mean(bvp_window[peaks])
            feats["bvp_pulse_amp_std"]  = np.std(bvp_window[peaks], ddof=1)
        else:
            for k in ["bvp_hr_bpm","bvp_ibi_mean_ms","bvp_ibi_std_ms",
                      "bvp_ibi_cv","bvp_pulse_amp_mean","bvp_pulse_amp_std"]:
                feats[k] = np.nan

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
            band_mask = (freqs >= 50) & (freqs <= 150)
            feats["emg_band_50_150_power"] = np.trapz(psd[band_mask], freqs[band_mask]) if band_mask.any() else 0.0

        # Activation bursts — threshold on rectified signal directly
        threshold = 3 * np.std(emg_rect)
        bursts, _ = find_peaks(emg_rect, height=threshold, distance=int(0.1 * fs))
        feats["emg_n_bursts"] = len(bursts)

    except Exception:
        pass

    return feats


def resp_features(resp_window, fs=FS):
    feats = {}
    feats.update(basic_stats(resp_window, "resp"))

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
            feats["resp_interval_std_s"]   = np.std(breath_intervals, ddof=1)
            feats["resp_irregularity_idx"] = feats["resp_interval_std_s"] / feats["resp_interval_mean_s"] if feats["resp_interval_mean_s"] > 0 else np.nan
        else:
            for k in ["resp_rate_bpm","resp_interval_mean_s",
                      "resp_interval_std_s","resp_irregularity_idx"]:
                feats[k] = np.nan

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
        else:
            feats["resp_dominant_freq"] = np.nan

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

    # Average COVAS score for the window
    row["covas_mean"] = np.nanmean(df_window["COVAS"].values)

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

def process_subject(filepath, subject_id):
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
    n_windows = n_samples // WINDOW_SAMPLES

    if n_windows == 0:
        print(f"    WARNING: subject {subject_id} has fewer than {WINDOW_SAMPLES} samples. Skipping.")
        return None

    rows = []
    for w in range(n_windows):
        start = w * WINDOW_SAMPLES
        end   = start + WINDOW_SAMPLES
        window_df = df.iloc[start:end]
        feats = extract_window_features(window_df, w)
        rows.append(feats)

    result_df = pd.DataFrame(rows)

    # Reorder: window metadata first, then features
    meta_cols = ["window_idx", "t_start_s", "t_end_s", "covas_mean"]
    feat_cols = [c for c in result_df.columns if c not in meta_cols]
    result_df = result_df[meta_cols + feat_cols]

    return result_df


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    csv_files = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".csv")
    ])

    if not csv_files:
        print(f"No CSV files found in '{INPUT_FOLDER}'. Check INPUT_FOLDER path.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in '{INPUT_FOLDER}'.\n")

    processed = 0
    for fname in csv_files:
        # Infer subject ID from filename (takes first integer found)
        import re
        nums = re.findall(r"\d+", fname)
        subject_id = int(nums[0]) if nums else fname

        filepath = os.path.join(INPUT_FOLDER, fname)
        result   = process_subject(filepath, subject_id)

        if result is not None:
            out_path = os.path.join(OUTPUT_FOLDER, f"subject_{subject_id:02d}_features.csv")
            result.to_csv(out_path, index=False)
            print(f"    Saved {len(result)} windows → {out_path}")
            processed += 1

    print(f"\nDone. {processed}/{len(csv_files)} subjects processed.")
    print(f"Feature CSVs written to '{OUTPUT_FOLDER}/'.")


if __name__ == "__main__":
    main()
