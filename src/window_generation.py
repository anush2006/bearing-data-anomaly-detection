import pandas as pd
import numpy as np
import os
from config import DATA_RAW_DIR,WINDOW_SIZE,STRIDE,DATA_EDA_DIR,DATA_PROCESSED_DIR


def load_label_data(path):
    """Load the CSV file containing filenames and labels."""
    df = pd.read_csv(path)
    return df

def load_raw_data(filepath):
    """Load raw signal data from one of the files."""
    data = []
    fp = open(filepath, "r")
    for line in fp:
        parts = line.strip().split()
        if len(parts) == 8:
            row = []
            for x in parts:
                row.append(float(x))
            data.append(row)
    fp.close()
    data = np.array(data, dtype=np.float32)     # (num_samples, 8)
    data = data.T                                # (8, num_samples)
    return data

def generate_windows(signal_data, window_size, stride):
    """Generate overlapping windows from the signal data."""
    windows = []
    num_channels, num_samples = signal_data.shape
    start = 0
    while start + window_size <= num_samples:
        end = start + window_size
        window = signal_data[:, start:end]       # (2, 2048)
        windows.append(window)
        start = start + stride
    return windows

def process_all_files(df, folder):
    """Core processing function to generate windows for each bearing."""
    healthy_b1 = []
    healthy_b2 = []
    healthy_b3 = []
    healthy_b4 = []

    faulty_b1 = []
    faulty_b2 = []
    faulty_b3 = []
    faulty_b4 = []

    for idx, row in df.iterrows():
        filename = row["filename"]
        label = row["label"]
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            continue

        full_signal = load_raw_data(filepath)    # (8, num_samples)
        """splitting for each bearing"""
        b1_signal = full_signal[0:2, :]          # (2, num_samples)
        b2_signal = full_signal[2:4, :]
        b3_signal = full_signal[4:6, :]
        b4_signal = full_signal[6:8, :]

        w1 = np.array(generate_windows(b1_signal, WINDOW_SIZE, STRIDE), dtype=np.float32)
        w2 = np.array(generate_windows(b2_signal, WINDOW_SIZE, STRIDE), dtype=np.float32)
        w3 = np.array(generate_windows(b3_signal, WINDOW_SIZE, STRIDE), dtype=np.float32)
        w4 = np.array(generate_windows(b4_signal, WINDOW_SIZE, STRIDE), dtype=np.float32)

        if label == 0:
            healthy_b1.append(w1)
            healthy_b2.append(w2)
            healthy_b3.append(w3)
            healthy_b4.append(w4)
        else:
            faulty_b1.append(w1)
            faulty_b2.append(w2)
            faulty_b3.append(w3)
            faulty_b4.append(w4)

    if len(healthy_b1) > 0: healthy_b1 = np.concatenate(healthy_b1, axis=0)
    else: healthy_b1 = np.array([])

    if len(healthy_b2) > 0: healthy_b2 = np.concatenate(healthy_b2, axis=0)
    else: healthy_b2 = np.array([])

    if len(healthy_b3) > 0: healthy_b3 = np.concatenate(healthy_b3, axis=0)
    else: healthy_b3 = np.array([])

    if len(healthy_b4) > 0: healthy_b4 = np.concatenate(healthy_b4, axis=0)
    else: healthy_b4 = np.array([])

    if len(faulty_b1) > 0: faulty_b1 = np.concatenate(faulty_b1, axis=0)
    else: faulty_b1 = np.array([])

    if len(faulty_b2) > 0: faulty_b2 = np.concatenate(faulty_b2, axis=0)
    else: faulty_b2 = np.array([])

    if len(faulty_b3) > 0: faulty_b3 = np.concatenate(faulty_b3, axis=0)
    else: faulty_b3 = np.array([])

    if len(faulty_b4) > 0: faulty_b4 = np.concatenate(faulty_b4, axis=0)
    else: faulty_b4 = np.array([])

    return healthy_b1, healthy_b2, healthy_b3, healthy_b4, faulty_b1, faulty_b2, faulty_b3, faulty_b4

def save_windows_to_npy(windows, filepath):
    """Save the generated windows to a .npy file."""
    np.save(filepath, windows)

if __name__ == "__main__":
    label_path = os.path.join(DATA_EDA_DIR, "labeled_test1_feature_table.csv")
    data_folder = os.path.join(DATA_RAW_DIR, "archive", "1st_test", "1st_test")

    df_labels = load_label_data(label_path)

    healthy_b1, healthy_b2, healthy_b3, healthy_b4, faulty_b1, faulty_b2, faulty_b3, faulty_b4 = process_all_files(df_labels, data_folder)

    save_windows_to_npy(healthy_b1, os.path.join(DATA_PROCESSED_DIR,"healthy_bearing1_windows.npy"))
    save_windows_to_npy(healthy_b2, os.path.join(DATA_PROCESSED_DIR,"healthy_bearing2_windows.npy"))
    save_windows_to_npy(healthy_b3, os.path.join(DATA_PROCESSED_DIR,"healthy_bearing3_windows.npy"))
    save_windows_to_npy(healthy_b4, os.path.join(DATA_PROCESSED_DIR,"healthy_bearing4_windows.npy"))
    save_windows_to_npy(faulty_b1, os.path.join(DATA_PROCESSED_DIR,"faulty_bearing1_windows.npy"))
    save_windows_to_npy(faulty_b2, os.path.join(DATA_PROCESSED_DIR,"faulty_bearing2_windows.npy"))
    save_windows_to_npy(faulty_b3, os.path.join(DATA_PROCESSED_DIR,"faulty_bearing3_windows.npy"))

    print(f"Healthy Bearing 1 Windows: {healthy_b1.shape}")
    print(f"Healthy Bearing 2 Windows: {healthy_b2.shape}")
    print(f"Healthy Bearing 3 Windows: {healthy_b3.shape}")
    print(f"Healthy Bearing 4 Windows: {healthy_b4.shape}")
    print(f"Faulty Bearing 1 Windows: {faulty_b1.shape}")
    print(f"Faulty Bearing 2 Windows: {faulty_b2.shape}")
    print(f"Faulty Bearing 3 Windows: {faulty_b3.shape}")
    print(f"Faulty Bearing 4 Windows: {faulty_b4.shape}")