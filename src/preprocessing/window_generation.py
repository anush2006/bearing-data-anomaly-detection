import pandas as pd
import numpy as np
import os
from config import DATA_RAW_DIR, WINDOW_SIZE, STRIDE, DATA_EDA_DIR, DATA_PROCESSED_DIR, CHANNELS_PER_BEARING

def load_label_data(path):
    df = pd.read_csv(path)
    return df

def load_raw_data(filepath):
    data = []
    with open(filepath, "r") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) == 8:
                row = [float(x) for x in parts]
                data.append(row)
    if len(data) == 0:
        return np.empty((8, 0), dtype=np.float32)
    data = np.array(data, dtype=np.float32)     # (num_samples, 8)
    data = data.T                                # (8, num_samples)
    return data

def generate_windows(signal_data, window_size, stride):
    windows = []
    if signal_data.size == 0:
        return np.empty((0, CHANNELS_PER_BEARING, window_size), dtype=np.float32)
    num_channels, num_samples = signal_data.shape
    start = 0
    while start + window_size <= num_samples:
        end = start + window_size
        window = signal_data[:, start:end]       # (2, window_size) expected
        windows.append(window.astype(np.float32))
        start = start + stride
    if len(windows) == 0:
        return np.empty((0, num_channels, window_size), dtype=np.float32)
    return np.array(windows, dtype=np.float32)

def process_all_files(df, folder):
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
        label = int(row["label"])
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            continue

        full_signal = load_raw_data(filepath)    # (8, num_samples)
        if full_signal.shape[0] != 8 or full_signal.shape[1] < WINDOW_SIZE:
            # skip files that do not have full 8 channels or are too short for one window
            continue

        b1_signal = full_signal[0:2, :]          # (2, num_samples)
        b2_signal = full_signal[2:4, :]
        b3_signal = full_signal[4:6, :]
        b4_signal = full_signal[6:8, :]

        w1 = generate_windows(b1_signal, WINDOW_SIZE, STRIDE)  # (N1,2,window)
        w2 = generate_windows(b2_signal, WINDOW_SIZE, STRIDE)
        w3 = generate_windows(b3_signal, WINDOW_SIZE, STRIDE)
        w4 = generate_windows(b4_signal, WINDOW_SIZE, STRIDE)

        if label == 0:
            if w1.shape[0] > 0: healthy_b1.append(w1)
            if w2.shape[0] > 0: healthy_b2.append(w2)
            if w3.shape[0] > 0: healthy_b3.append(w3)
            if w4.shape[0] > 0: healthy_b4.append(w4)
        else:
            if w1.shape[0] > 0: faulty_b1.append(w1)
            if w2.shape[0] > 0: faulty_b2.append(w2)
            if w3.shape[0] > 0: faulty_b3.append(w3)
            if w4.shape[0] > 0: faulty_b4.append(w4)

    def concat_or_empty(list_of_arrays):
        if len(list_of_arrays) == 0:
            return np.empty((0, CHANNELS_PER_BEARING, WINDOW_SIZE), dtype=np.float32)
        return np.concatenate(list_of_arrays, axis=0)

    healthy_b1 = concat_or_empty(healthy_b1)
    healthy_b2 = concat_or_empty(healthy_b2)
    healthy_b3 = concat_or_empty(healthy_b3)
    healthy_b4 = concat_or_empty(healthy_b4)

    faulty_b1 = concat_or_empty(faulty_b1)
    faulty_b2 = concat_or_empty(faulty_b2)
    faulty_b3 = concat_or_empty(faulty_b3)
    faulty_b4 = concat_or_empty(faulty_b4)

    return healthy_b1, healthy_b2, healthy_b3, healthy_b4, faulty_b1, faulty_b2, faulty_b3, faulty_b4

def save_windows_to_npy(windows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
    save_windows_to_npy(faulty_b4, os.path.join(DATA_PROCESSED_DIR,"faulty_bearing4_windows.npy"))

    print(f"Healthy Bearing 1 Windows: {healthy_b1.shape}")
    print(f"Healthy Bearing 2 Windows: {healthy_b2.shape}")
    print(f"Healthy Bearing 3 Windows: {healthy_b3.shape}")
    print(f"Healthy Bearing 4 Windows: {healthy_b4.shape}")
    print(f"Faulty Bearing 1 Windows: {faulty_b1.shape}")
    print(f"Faulty Bearing 2 Windows: {faulty_b2.shape}")
    print(f"Faulty Bearing 3 Windows: {faulty_b3.shape}")
    print(f"Faulty Bearing 4 Windows: {faulty_b4.shape}")
