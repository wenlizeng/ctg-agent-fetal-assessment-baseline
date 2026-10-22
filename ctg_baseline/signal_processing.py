# -*- coding: utf-8 -*-
"""Signal processing utilities shared across tasks."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.interpolate import interp1d


# ===========================================================================
# Filtering
# ===========================================================================

def butter_bandpass(lowcut, highcut, fs, order=2):
    """Design Butterworth bandpass filter (sos form)."""
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq],
                  btype='bandpass', output='sos')


def DUS_filtering(signal, fs, lowcut=50.0, highcut=600.0, order=4):
    """
    Apply bandpass filter to Doppler Ultrasound signal with adaptive cutoff.

    Cutoff frequencies are dynamically clamped below the Nyquist frequency
    (fs / 2) so the filter works correctly regardless of the input sampling
    rate (e.g. 4000 Hz, 1000 Hz, or any other rate).

    Parameters
    ----------
    signal : np.ndarray — raw DUS signal
    fs : int — sampling rate in Hz
    lowcut : float — desired low-cut frequency (Hz)
    highcut : float — desired high-cut frequency (Hz)
    order : int — Butterworth filter order

    Returns
    -------
    filtered_signal : np.ndarray
    """
    nyq = 0.5 * fs

    # Clamp cutoffs to safe range below Nyquist
    actual_lowcut = min(lowcut, nyq - 2.0)
    actual_highcut = min(highcut, nyq - 1.0)

    # Ensure lowcut < highcut
    if actual_lowcut >= actual_highcut:
        actual_lowcut = actual_highcut / 2.0

    # Normalise to (0, 1) for scipy
    low = actual_lowcut / nyq
    high = actual_highcut / nyq

    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, signal)


# ===========================================================================
# Resampling
# ===========================================================================

def signal_resample(signal, original_fs, target_fs):
    """Polyphase resample signal to target sampling rate."""
    from math import gcd
    g = gcd(int(original_fs), int(target_fs))
    up = int(target_fs) // g
    down = int(original_fs) // g
    return resample_poly(signal, up, down)


# ===========================================================================
# Scalogram
# ===========================================================================

def create_scalogram_from_signal(sig, fs, time_bins, freq_bins):
    """CWT scalogram using Morlet wavelet."""
    import pywt
    from skimage.transform import resize

    scales = np.arange(1, freq_bins + 1)
    coeffs, _ = pywt.cwt(sig, scales, wavelet='morl',
                         sampling_period=1.0 / fs)
    scalogram = resize(np.abs(coeffs), (freq_bins, time_bins),
                       mode='constant')
    return scalogram


# ===========================================================================
# FHR Signal Processing (Tasks 2, 3, 4)
# ===========================================================================

def interpolate_fhr(fhr, max_gap_sec=15, fs=4):
    """
    Interpolate missing FHR values (encoded as 0 or NaN).

    Parameters
    ----------
    fhr : np.ndarray — raw FHR
    max_gap_sec : int — maximum interpolatable gap in seconds
    fs : float — sampling rate in Hz

    Returns
    -------
    fhr_interp : np.ndarray
    quality_mask : np.ndarray (bool) — True = valid
    """
    max_gap_samples = int(max_gap_sec * fs)
    fhr = fhr.copy().astype(np.float32)
    fhr[fhr == 0] = np.nan

    quality_mask = ~np.isnan(fhr)
    if not quality_mask.any():
        return fhr, quality_mask

    # Linear interpolation for short gaps
    x = np.arange(len(fhr))
    valid_x = x[quality_mask]
    valid_y = fhr[quality_mask]

    fhr_interp = interp1d(valid_x, valid_y, kind='linear',
                          bounds_error=False, fill_value='extrapolate')(x)

    # Create quality mask: invalidate gaps > max_gap_sec
    gap_start = None
    for i in range(len(fhr)):
        if np.isnan(fhr[i]) or fhr[i] <= 0:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_len = i - gap_start
                if gap_len > max_gap_samples:
                    quality_mask[gap_start:i] = False
                gap_start = None

    if gap_start is not None:
        gap_len = len(fhr) - gap_start
        if gap_len > max_gap_samples:
            quality_mask[gap_start:] = False

    fhr_interp[~quality_mask] = np.nan
    return fhr_interp, quality_mask


def detect_baseline(fhr, window_sec=600, fs=4):
    """
    Estimate FHR baseline using sliding window median filter.

    Parameters
    ----------
    fhr : np.ndarray
    window_sec : int — window size in seconds (default 10 min)
    fs : float

    Returns baseline : np.ndarray
    """
    from scipy.ndimage import median_filter
    window = int(window_sec * fs)
    if window % 2 == 0:
        window += 1
    fhr_filled = np.where(np.isnan(fhr), np.nanmedian(fhr), fhr)
    return median_filter(fhr_filled, size=window)


def detect_accelerations_decelerations(fhr, baseline, fs=4,
                                        amp_thresh=15, min_dur=15):
    """
    Detect accelerations and decelerations relative to baseline.

    Returns
    -------
    seg : np.ndarray — official encoding: 0=baseline, 1=accel, -1=decel
    """
    min_samples = int(min_dur * fs)
    diff = fhr - baseline
    seg = np.zeros(len(fhr), dtype=np.int32)

    # Identify regions exceeding threshold
    accel_mask = diff >= amp_thresh
    decel_mask = diff <= -amp_thresh

    # Filter short segments
    def _filter_short(mask, label_val):
        from scipy.ndimage import label
        labeled, n = label(mask)
        for i in range(1, n + 1):
            region = labeled == i
            if region.sum() < min_samples:
                continue
            seg[region] = label_val

    _filter_short(accel_mask, 1)
    _filter_short(decel_mask, -1)

    return seg


# ===========================================================================
# Feature Extraction for Task 3/4
# ===========================================================================

def extract_ctg_features(fhr, uc, fs=4):
    """
    Extract hand-crafted CTG features for classification.

    Returns dict of scalar features.
    """
    fhr_valid = fhr[~np.isnan(fhr)]
    uc_valid = uc[~np.isnan(uc)]

    features = {}

    # FHR statistics
    features['fhr_mean'] = np.nanmean(fhr_valid)
    features['fhr_std'] = np.nanstd(fhr_valid)
    features['fhr_min'] = np.nanmin(fhr_valid)
    features['fhr_max'] = np.nanmax(fhr_valid)
    features['fhr_missing_pct'] = np.isnan(fhr).mean()

    # UC statistics
    features['uc_mean'] = np.nanmean(uc_valid)
    features['uc_std'] = np.nanstd(uc_valid)
    features['uc_missing_pct'] = np.isnan(uc).mean()

    # Baseline + variability
    baseline = detect_baseline(fhr, window_sec=600, fs=fs)
    features['baseline_mean'] = np.nanmean(baseline)

    fhr_diff = np.diff(fhr_valid)
    features['fhr_rmssd'] = np.sqrt(np.nanmean(fhr_diff ** 2))

    # Acceleration/deceleration counts
    seg = detect_accelerations_decelerations(fhr, baseline, fs=fs)
    # Count transitions
    accel_count = 0
    decel_count = 0
    in_accel = False
    in_decel = False
    for s in seg:
        if s == 1 and not in_accel:
            accel_count += 1
            in_accel = True
            in_decel = False
        elif s == -1 and not in_decel:
            decel_count += 1
            in_decel = True
            in_accel = False
        elif s == 0:
            in_accel = False
            in_decel = False
    features['n_accelerations'] = accel_count
    features['n_decelerations'] = decel_count

    return features
