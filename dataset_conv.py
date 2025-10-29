import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import train_test_split

# Path to WFDB files
DATA_DIR = "E:\Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"

# Parameters
SAMPLES_PER_BEAT = 187
LEAD_NAME = "MLII"
LABEL_MAP = {
    'N': 0,  # Normal
    'V': 1,  # Ventricular ectopic
    'A': 2,  # Atrial ectopic
    'L': 3,  # Left bundle branch block (example)
    'R': 4   # Right bundle branch block (example)
}

def extract_beats(record_path):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    if LEAD_NAME not in record.sig_name:
        return None, None

    lead_idx = record.sig_name.index(LEAD_NAME)
    signal = record.p_signal[:, lead_idx]

    X_beats, y_labels = [], []

    for sample, sym in zip(ann.sample, ann.symbol):
        if sym not in LABEL_MAP:
            continue
        start = sample - 90
        end = sample + 97  # 90 before + 97 after = 187 samples
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        beat_resampled = resample(beat, SAMPLES_PER_BEAT)
        X_beats.append(beat_resampled)
        y_labels.append(LABEL_MAP[sym])

    return np.array(X_beats), np.array(y_labels)

# Collect all beats from all records
all_X, all_y = [], []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".dat"):
        record_name = fname[:-4]
        record_path = os.path.join(DATA_DIR, record_name)
        X, y = extract_beats(record_path)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

all_X = np.vstack(all_X)
all_y = np.hstack(all_y)

# Combine into DataFrame like MIT-BIH CSV
df = pd.DataFrame(all_X)
df['label'] = all_y

# Split into train/test and save as CSV
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df.to_csv("mitbih_train.csv", header=False, index=False)
test_df.to_csv("mitbih_test.csv", header=False, index=False)

print("✅ Saved mitbih_train.csv and mitbih_test.csv")
