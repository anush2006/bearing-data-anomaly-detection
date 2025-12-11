import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from config import DATA_EDA_DIR
df = pd.read_csv(os.path.join(DATA_EDA_DIR,"test1_feature_table.csv"))
plt.figure(figsize=(14, 8))

for ch in range(1, 9):
    plt.plot(df["timestamp_min"], df[f"ch{ch}_kurtosis"], label=f"ch{ch}_kurtosis")

plt.title("Kurtosis Trend Across All Bearings")
plt.xlabel("Time (minutes)")
plt.ylabel("Kurtosis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

for ch in range(1, 9):
    plt.plot(df["timestamp_min"], df[f"ch{ch}_rms"], label=f"ch{ch}_kurtosis")

plt.title("RMS Trend Across All Bearings")
plt.xlabel("Time (minutes)")
plt.ylabel("RMS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
