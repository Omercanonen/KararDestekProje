import pandas as pd

DROP_ALWAYS = [
    "encounter_id", "patient_nbr", "weight", "payer_code"
]

# İlaçlardan tek değerli olanlar burada tutulacak (run-time'da eklenecek)
POTENTIAL_CONST_DRUGS = [
    "examide", "acetohexamide", "tolbutamide", "troglitazone"
]

A1C_MAP = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
GLU_MAP = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}

DRUG_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 2}

def to_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["readmit_30"] = df["readmitted"].replace({"<30":1, "NO":0, ">30":0}).astype(int)
    df = df.drop(columns=["readmitted"])
    return df

def drop_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Discharge disposition: ölüm/hospice kayıtlarını çıkar
    # 11 (Expired), 19, 20, 21 (Hospice)
    if "discharge_disposition_id" in df.columns:
        bad = {11, 19, 20, 21}
        df = df[~df["discharge_disposition_id"].isin(bad)]

    # 2) Gender: Unknown/Invalid kayıtları çıkar
    if "gender" in df.columns:
        df = df[df["gender"].isin(["Male", "Female"])]

    # 3) Sabit (kimlik/çok eksik) sütunları at
    drop_cols = [c for c in DROP_ALWAYS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 4) Potansiyel tek-değerli ilaç sütunlarını tespit edip at
    const_drugs = []
    for c in POTENTIAL_CONST_DRUGS:
        if c in df.columns and df[c].nunique(dropna=False) <= 1:
            const_drugs.append(c)
    if const_drugs:
        df = df.drop(columns=const_drugs, errors="ignore")

    return df

def basic_impute_and_maps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Kategorik eksikler
    for col in ["race", "medical_specialty"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # age: "[70-80)" -> 75 (orta nokta)
    if "age" in df.columns:
        def age_to_mid(a):
            # beklenen format: "[10-20)"
            try:
                a = a.strip("[]()")
                lo, hi = a.split("-")
                return (int(lo) + int(hi)) / 2
            except Exception:
                return None
        df["age_mid"] = df["age"].apply(age_to_mid)

    # A1C / Max Glucose
    if "A1Cresult" in df.columns:
        df["A1Cresult_ord"] = df["A1Cresult"].map(A1C_MAP).fillna(0).astype(int)
    if "max_glu_serum" in df.columns:
        df["max_glu_serum_ord"] = df["max_glu_serum"].map(GLU_MAP).fillna(0).astype(int)

    # İlaç sütunlarını ordinal'a çevir
    drug_cols = [
        c for c in df.columns
        if c not in ["change", "diabetesMed"] and
           df[c].dtype == "object" and
           set(df[c].dropna().unique()).issubset({"No","Steady","Up","Down"})
    ]
    for c in drug_cols:
        df[c + "_ord"] = df[c].map(DRUG_MAP).fillna(0).astype(int)

    # change / diabetesMed
    if "change" in df.columns:
        df["change_bin"] = df["change"].map({"Ch":1, "No":0}).fillna(0).astype(int)
    if "diabetesMed" in df.columns:
        df["diabetesMed_bin"] = df["diabetesMed"].map({"Yes":1, "No":0}).fillna(0).astype(int)

    return df
