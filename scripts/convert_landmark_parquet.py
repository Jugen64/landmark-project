import pandas as pd
from pathlib import Path
import csv

RAW_ROOT = Path("~/Documents/Code/projects/datasets/landmark_to_country").expanduser()
PARQUET_PATH = RAW_ROOT / "landmark_to_country.parquet"
CSV_OUT_PATH = RAW_ROOT / "landmark_to_country.csv"



def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(df.columns)
    print(len(df))

    df = df[["id", "country"]]
    df.to_csv(CSV_OUT_PATH, index=False)

if __name__ == "__main__":
    main()