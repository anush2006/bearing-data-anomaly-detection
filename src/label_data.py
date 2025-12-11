import pandas as pd
import os
from config import DATA_EDA_DIR
path=os.path.join(DATA_EDA_DIR,"test1_feature_table.csv")
fault_start = 1500  # in minutes(approximation from EDA results)

def load_label_data(path):
    """Function to load label data from a CSV file."""
    df = pd.read_csv(path)
    return df

def add_labels(df):
    """Function to add labels to the DataFrame based on filename patterns."""
    df['label'] = 0  # Default label for normal data
    for idx, row in df.iterrows():
        timestamp = row['timestamp_min']
        
        if idx >= fault_start:
            df.at[idx, 'label'] = 1  # Faulty data
    return df

def save_labeled_data(df, path=os.path.join(DATA_EDA_DIR,"labeled_test1_feature_table.csv")):
    """Function to save the labeled DataFrame to a CSV file."""
    df.to_csv(path, index=False)
    print(f"Labeled data saved to {path}")

if __name__ == "__main__":
    df = load_label_data(path)
    df_labeled = add_labels(df)
    save_labeled_data(df_labeled)