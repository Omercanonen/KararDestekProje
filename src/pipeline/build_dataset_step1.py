import os
import pandas as pd
from src.preprocessing.load import load_raw_diabetes
from src.preprocessing.clean import (
    to_label, drop_and_filter, basic_impute_and_maps
)

RAW_CSV = "data/diabetic_data.csv"
INTERIM_DIR = "data/interim"
os.makedirs(INTERIM_DIR, exist_ok=True)

def main():
    df = load_raw_diabetes(RAW_CSV)
    print("Orijinal şekil:", df.shape)

    df = to_label(df)
    df = drop_and_filter(df)
    df = basic_impute_and_maps(df)

    # İlk ara çıktı (özellik kodlama/split henüz yok)
    out_csv = os.path.join(INTERIM_DIR, "clean_step1.csv")
    df.to_csv(out_csv, index=False)
    print("Kaydedildi:", out_csv)
    print("Yeni şekil:", df.shape)

if __name__ == "__main__":
    main()
