from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DAILY_LOGS_DIR = PROJECT_ROOT / "daily_logs"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EDA_DIR = DATA_DIR / "EDA"
EDA_ANALYSIS_DIR = PROJECT_ROOT / "EDA_analysis"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SRC_DIR = PROJECT_ROOT / "src"
VENV_DIR = PROJECT_ROOT / "venv"

WINDOW_SIZE = 2048
STRIDE = 256

NUM_BEARINGS = 4
CHANNELS_PER_BEARING = 2

def ensure_dirs():
    dirs = [
        DAILY_LOGS_DIR,
        DATA_DIR,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_EDA_DIR,
        EDA_ANALYSIS_DIR,
        OUTPUTS_DIR,
        SRC_DIR,
        VENV_DIR
    ]

    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Could not create directory {d}: {e}")

if __name__ == "__main__":
    ensure_dirs()
    print("All required directories ensured.")
