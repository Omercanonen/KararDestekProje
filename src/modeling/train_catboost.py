import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

# Veri yolları
DATA_PATH = "data/processed/train.csv"
VALID_PATH = "data/processed/valid.csv"
MODEL_DIR = "models/catboost"
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------- FEATURE ENGINEERING -------------------
def feature_engineering(df):
    """Yeni anlamlı özellikler ekler ve kategorik dönüşümler yapar."""
    df["hospital_visits_total"] = (
        df["number_inpatient"] + df["number_emergency"] + df["number_outpatient"]
    )
    df["long_stay_flag"] = (df["time_in_hospital"] > 7).astype(int)
    df["medication_density"] = df["num_medications"] / (df["num_lab_procedures"] + 1)
    df["age_mid"] = df["age"].replace({
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    })
    return df


# ------------------- MAIN TRAINING -------------------
def main():
    print("📥 Veri yükleniyor...")
    train_df = pd.read_csv(DATA_PATH)
    valid_df = pd.read_csv(VALID_PATH)

    # Özellik mühendisliği
    X_train = feature_engineering(train_df.drop(columns=["readmit_30"]))
    y_train = train_df["readmit_30"]
    X_valid = feature_engineering(valid_df.drop(columns=["readmit_30"]))
    y_valid = valid_df["readmit_30"]

    # Kategorik sütunları bul
    cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    # ✅ Eksik ve karışık tipli kategorik değerleri düzelt
    X_train[cat_features] = X_train[cat_features].fillna("Unknown")
    X_valid[cat_features] = X_valid[cat_features].fillna("Unknown")

    for col in cat_features:
        X_train[col] = X_train[col].astype(str)
        X_valid[col] = X_valid[col].astype(str)

    # Ölçekleme sadece sayısal sütunlara uygulanır
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid[num_cols] = scaler.transform(X_valid[num_cols])

    # CatBoost modeli
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=200
    )

    print("🚀 Model eğitiliyor...")
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Tahminler
    y_pred_proba = model.predict_proba(X_valid)[:, 1]

    # 🔧 En iyi olasılık eşiğini ROC eğrisine göre bul
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)
    best_idx = (tpr - fpr).argmax()
    best_threshold = thresholds[best_idx]
    print(f"🔧 En iyi olasılık eşiği: {best_threshold:.2f}")

    y_pred = (y_pred_proba > best_threshold).astype(int)

    # 📊 Performans metrikleri
    metrics = {
        "roc_auc": roc_auc_score(y_valid, y_pred_proba),
        "accuracy": accuracy_score(y_valid, y_pred),
        "f1": f1_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred),
        "recall": recall_score(y_valid, y_pred)
    }

    print("\n📊 Model Performansı:")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")

    # 💾 Model ve metrikleri kaydet
    joblib.dump(model, os.path.join(MODEL_DIR, "catboost_model.joblib"))
    pd.DataFrame([metrics]).to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)

    print(f"\n💾 Model kaydedildi: {MODEL_DIR}/catboost_model.joblib")


if __name__ == "__main__":
    main()
