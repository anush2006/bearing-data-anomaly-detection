import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis

def rms(x):
    return np.sqrt(np.mean(x * x))

folder = 'archive/1st_test/1st_test'

def load_bearing_file(path):
    """Function for loading a bearing data file.
    Each file contains 8 columns of float data separated by spaces."""
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                row = [float(x) for x in parts]
                data.append(row)
            else:
                continue
    return data


records = []

for idx, fname in enumerate(sorted(os.listdir(folder))):
    fpath = os.path.join(folder, fname)

    # Load raw signal (8 columns)
    raw = load_bearing_file(fpath)
    df_raw = pd.DataFrame(raw)  # shape ~ (20000, 8)

    row = {
        "file_index": idx,
        "timestamp_min": idx * 10,
        "filename": fname
    }

    # Extract features for 8 channels
    for ch in range(8):
        col = df_raw[ch].values
        
        row[f"ch{ch+1}_rms"]       = rms(col)
        row[f"ch{ch+1}_kurtosis"]  = kurtosis(col, fisher=False)

    records.append(row)
    print(f"Processed file {idx+1}/{len(os.listdir(folder))}: {fname}")

df_final = pd.DataFrame(records)
df_final.to_csv("test1_feature_table.csv", index=False)

print(df_final.head())