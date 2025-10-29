import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =============================
# Configuration
# =============================
DATA_DIR = "."  # Folder where mitbih_train.csv and mitbih_test.csv are stored
NUM_CLIENTS = 5
SPLIT_TYPE = "iid"   # Options: "iid", "non_iid", "patient"
BASE_OUTPUT_DIR = "./federated_clients"

# Create subfolder for chosen split type
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, SPLIT_TYPE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Load Dataset
# =============================
train_df = pd.read_csv(os.path.join(DATA_DIR, "mitbih_train.csv"), header=None)
test_df = pd.read_csv(os.path.join(DATA_DIR, "mitbih_test.csv"), header=None)

# Last column = label
train_df.columns = [f"f{i}" for i in range(train_df.shape[1] - 1)] + ["label"]
test_df.columns = [f"f{i}" for i in range(test_df.shape[1] - 1)] + ["label"]

print(f"✅ Loaded train: {train_df.shape}, test: {test_df.shape}")

# =============================
# Helper Functions
# =============================
def log_class_distribution(df, client_id):
    counts = df['label'].value_counts().to_dict()
    print(f"Client {client_id} class distribution: {counts}")

def create_iid_splits(df):
    """Evenly random split (IID)."""
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    splits = np.array_split(df, NUM_CLIENTS)
    print("📊 IID Split: evenly distributed samples across clients")
    return splits

def create_non_iid_splits(df):
    """Class-based non-IID: each client gets majority from one or two classes."""
    splits = []
    classes = df['label'].unique()
    np.random.shuffle(classes)

    for i in range(NUM_CLIENTS):
        class_subset = classes[i % len(classes)]
        client_data_major = df[df['label'] == class_subset]

        # Handle case where a class may be small or empty
        if len(client_data_major) == 0:
            print(f"⚠️ Skipping major class {class_subset} for client {i+1} (no samples found)")
            client_data_major = df.sample(frac=0.1, random_state=i)

        # Add small random portion from other classes for realism
        client_data_minor = df[df['label'] != class_subset]
        if len(client_data_minor) > 0:
            client_data_minor = client_data_minor.sample(frac=0.05, random_state=i, replace=True)
        else:
            client_data_minor = pd.DataFrame(columns=df.columns)

        client_df = pd.concat([client_data_major, client_data_minor])
        client_df = client_df.sample(frac=1, random_state=i).reset_index(drop=True)
        splits.append(client_df)

        print(f"Client {i+1}: major class {class_subset}, total {len(client_df)} samples")
        log_class_distribution(client_df, i + 1)

    print("📊 Non-IID Split: each client biased toward specific classes")
    return splits

def create_patient_splits(df):
    """
    Patient-based split (simulated).
    Since MIT-BIH CSV doesn’t contain patient IDs, we simulate pseudo-patients
    by grouping beats sequentially into chunks.
    """
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    chunk_size = len(df) // NUM_CLIENTS
    splits = [df.iloc[i*chunk_size : (i+1)*chunk_size].copy() for i in range(NUM_CLIENTS)]

    print("📊 Patient-based Split: data divided sequentially (simulated patients)")
    return splits

# =============================
# Split Logic
# =============================
if SPLIT_TYPE == "iid":
    client_splits = create_iid_splits(train_df)
elif SPLIT_TYPE == "non_iid":
    client_splits = create_non_iid_splits(train_df)
elif SPLIT_TYPE == "patient":
    client_splits = create_patient_splits(train_df)
else:
    raise ValueError("Invalid SPLIT_TYPE. Choose from: iid, non_iid, patient")

# =============================
# Save Client Data
# =============================
for i, client_df in enumerate(client_splits):
    if client_df.empty:
        print(f"⚠️ Warning: Client {i+1} has no data, skipping save.")
        continue
    path = os.path.join(OUTPUT_DIR, f"client_{i+1}_train.csv")
    client_df.to_csv(path, index=False)
    print(f"💾 Saved {path} ({len(client_df)} samples)")

# Split test set equally among clients for evaluation
test_splits = np.array_split(test_df.sample(frac=1, random_state=42).reset_index(drop=True), NUM_CLIENTS)
for i, client_df in enumerate(test_splits):
    path = os.path.join(OUTPUT_DIR, f"client_{i+1}_test.csv")
    client_df.to_csv(path, index=False)

print("\n✅ Federated client datasets created successfully!")
print(f"📂 Location: {os.path.abspath(OUTPUT_DIR)}")
