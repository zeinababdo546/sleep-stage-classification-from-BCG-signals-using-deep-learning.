import os
import mne
import numpy as np
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
from mne.preprocessing.ecg import find_ecg_events
from scipy.signal import welch, medfilt
from scipy.integrate import trapezoid

# --- 1. Processing Functions ---
def get_hrv_features(window, fs=4.0):
    freqs, psd = welch(window, fs=fs, nperseg=128)
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    
    lf = trapezoid(psd[lf_mask], freqs[lf_mask])
    hf = trapezoid(psd[hf_mask], freqs[hf_mask])
    
    total = lf + hf
    ratio = lf / hf if hf > 0 else 0
    lf_nu = (lf / total * 100) if total > 0 else 0
    hf_nu = (hf / total * 100) if total > 0 else 0
    
    return [lf, hf, lf_nu, hf_nu, ratio]

def parse_annotations(xml_path, mapping):
    tree = ET.parse(xml_path)
    events = []
    for ev in tree.findall('.//ScoredEvent'):
        concept = ev.find('EventConcept').text
        if concept in mapping:
            events.append({
                'label': mapping[concept],
                'start': float(ev.find('Start').text),
                'dur': float(ev.find('Duration').text)
            })
    return events

# --- 2. Settings ---
paths = {
    "signals": r'C:\Users\compu_tech\OneDrive\Desktop\Signals\sleepproject\Dataset\Signals',
    "annots": r'C:\Users\compu_tech\OneDrive\Desktop\Signals\sleepproject\Dataset\Annotations'
}

mapping = {
    'Wake|0': 0,
    'Stage 1 sleep|1': 1,
    'Stage 2 sleep|2': 2,
    'Stage 3 sleep|3': 3,
    'REM sleep|5': 4
}

signals_list, features_list, labels_list = [], [], []
file_ids = [f.replace('.edf', '') for f in os.listdir(paths["signals"]) if f.endswith('.edf')]

# --- 3. Pipeline ---
for fid in file_ids:
    try:
        # A- Load signal using modern .pick()
        raw = mne.io.read_raw_edf(
            os.path.join(paths["signals"], f"{fid}.edf"),
            preload=False,
            verbose=False
        )

        raw.pick(['EKG']).load_data()
        raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)

        # B- Heartbeat detection
        events, _, _ = find_ecg_events(raw, ch_name='EKG', verbose=False)
        fs_orig = raw.info['sfreq']

        # Ensure array is 1D using .flatten()
        peaks = events[:, 0].flatten()
        ibi = np.diff(peaks) / fs_orig

        mask = (ibi > 0.33) & (ibi < 1.5)

        # Fix error: ensure times and ihr are 1D arrays
        times_raw = (peaks[1:][mask] / fs_orig).flatten()
        ihr_cleaned = medfilt(60.0 / ibi[mask], kernel_size=3).flatten()

        if len(ihr_cleaned) < 50:
            continue

        # C- Interpolation with valid input dimensions
        f_interp = interp1d(
            times_raw,
            ihr_cleaned,
            kind='slinear',
            fill_value="extrapolate"
        )

        new_times = np.arange(times_raw[0], times_raw[-1], 0.25)
        ihr_4hz = f_interp(new_times)

        # D- Segmentation and feature extraction
        scored_events = parse_annotations(
            os.path.join(paths["annots"], f"{fid}-nsrr.xml"),
            mapping
        )

        for ev in scored_events:
            for w in range(int(ev['dur'] // 30)):
                curr_start = ev['start'] + (w * 30)
                idx = int((curr_start - times_raw[0]) * 4)

                if idx >= 0 and (idx + 120) <= len(ihr_4hz):
                    window = ihr_4hz[idx:idx+120]

                    signals_list.append(
                        (window - np.mean(window)) /
                        (np.std(window) + 1e-6)
                    )

                    features_list.append(get_hrv_features(window))
                    labels_list.append(ev['label'])

    except Exception as e:
        print(f" Error in {fid}: {e}")

# Final conversion
X_signals = np.expand_dims(
    np.array(signals_list, dtype='float32'),
    axis=-1
)

X_features = np.array(features_list, dtype='float32')
Y = np.array(labels_list, dtype='int64')

# 1. Signals (X_signals) - use float32 to reduce RAM usage
X_final_signals = np.array(X_signals, dtype='float32')

# 2. Features (X_features) - use float32
X_final_features = np.array(X_features, dtype='float32')

# 3. Labels (Y) - use int64 (important for SMOTE and CrossEntropy)
Y_final_labels = np.array(Y, dtype='int64')

np.save('X2_data.npy', X_final_signals)       # Windowed & normalized signals
np.save('X2_features.npy', X_final_features)  # 5 HRV features (LF, HF, etc.)
np.save('Y2_labels.npy', Y_final_labels)      # True labels (0, 1, 2, 3, 4)

print(f"Signals Shape: {X_final_signals.shape}")
print(f"Features Shape: {X_final_features.shape}")
print(f"Labels Shape: {Y_final_labels.shape}")
print(f"Unique Labels: {np.unique(Y_final_labels)}")
```
