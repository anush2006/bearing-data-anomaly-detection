import numpy as np
import torch
from torch.utils.data import Dataset

class BearingDataset(Dataset):
    """
    Dataset for loading windowed vibration data for all bearings.
    Each item returned is (window_tensor, bearing_id).
    """
    def __init__(self, healthy_paths, transform=None):
        """
        healthy_paths: dict mapping bearing_id -> npy filepath
                       Example: {1: "...b1.npy", 2: "...b2.npy", ...}
        """
        self.data = []
        self.transform = transform
        for bearing_id, path in healthy_paths.items():
            windows = np.load(path)  # shape (N, 2, 2048)
            for w in windows:
                self.data.append((w, bearing_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        window, bearing_id = self.data[index]
        window_tensor = torch.tensor(window, dtype=torch.float32)
        bearing_id = torch.tensor(bearing_id, dtype=torch.long)

        return window_tensor, bearing_id
