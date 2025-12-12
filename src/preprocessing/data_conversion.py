import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis
from config import DATA_RAW_DIR,DATA_EDA_DIR

def rms(x):
    """Function to compute Root Mean Square (RMS) of a 1D array."""
    return np.sqrt(np.mean(x * x))

folder = os.path.join(DATA_RAW_DIR, "archive", "1st_test", "1st_test")

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
    """Process each bearing data file to extract features."""
    fpath = os.path.join(folder, fname)
    raw = load_bearing_file(fpath)
    df_raw = pd.DataFrame(raw) 

    row = {
        "file_index": idx,
        "timestamp_min": idx * 10,
        "filename": fname
    }
    for ch in range(8):
        col = df_raw[ch].values
        
        row[f"ch{ch+1}_rms"]       = rms(col)
        row[f"ch{ch+1}_kurtosis"]  = kurtosis(col, fisher=False)

    records.append(row)
    print(f"Processed file {idx+1}/{len(os.listdir(folder))}: {fname}")

df_final = pd.DataFrame(records)
df_final.to_csv(os.path.join(DATA_EDA_DIR,"test1_feature_table.csv"), index=False)

print(df_final.head(15))