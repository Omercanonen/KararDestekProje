import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from src.preprocessing.diag_map import add_diag_categories

INPUT_CSV = "data/interim/clean_step1.csv"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def label_encode_columns(df, cols):
    encoders = {}
    for c in cols:
        le = LabelEncoder()
        df[c] = df[c].astype(str)
        df[c] = le.fit_transform(df[c])
        encoders[c] = le
    return df, encoders

def main():
    df = pd.read_csv(INPUT_CSV)
    print("Yüklenen veri:", df.shape)

    # 1) ICD kategorileri
    df = add_diag_categories(df)

    # 2) Kategorik sütun listesi
    cat_cols = [
        "race", "gender", "medical_specialty",
        "admission_type_id", "admission_source_id", "discharge_disposition_id",
        "diag_1_cat", "diag_2_cat", "diag_3_cat"
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    df, encoders = label_encode_columns(df, cat_cols)

    # 3) Train/Valid/Test böl
    X = df.drop(columns=["readmit_30"])
    y = df["readmit_30"]

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(sss1.split(X, y))
    X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
    y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    valid_idx, test_idx = next(sss2.split(X_temp, y_temp))
    X_valid, X_test = X_temp.iloc[valid_idx], X_temp.iloc[test_idx]
    y_valid, y_test = y_temp.iloc[valid_idx], y_temp.iloc[test_idx]

    # 4) Kaydet
    for name, X_part, y_part in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test)
    ]:
        df_part = X_part.copy()
        df_part["readmit_30"] = y_part
        out_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        df_part.to_csv(out_path, index=False)
        print(f"{name} → {df_part.shape} → {out_path}")

if __name__ == "__main__":
    main()
