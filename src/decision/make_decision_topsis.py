
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # src klasöründen yukarı çık
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost", "catboost_model.joblib")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "reports", "recommendations", "topsis_decision.csv")


os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def topsis(matrix, weights, impacts):
    # Sütun varyansı 0 olanları ele
    valid_cols = np.where(matrix.std(axis=0) > 1e-9)[0]
    if len(valid_cols) == 0:
        raise ValueError("Tüm sütunlar sabit, TOPSIS uygulanamaz.")
    matrix = matrix[:, valid_cols]
    weights = weights[valid_cols]
    impacts = [impacts[i] for i in valid_cols]

    # Normalize et
    denom = np.sqrt((matrix ** 2).sum(axis=0))
    denom[denom == 0] = 1e-9
    norm = matrix / denom
    norm = np.nan_to_num(norm, nan=0.0)

    weighted = norm * weights
    ideal_best = np.where(np.array(impacts) == '+', weighted.max(axis=0), weighted.min(axis=0))
    ideal_worst = np.where(np.array(impacts) == '+', weighted.min(axis=0), weighted.max(axis=0))
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = dist_worst / (dist_best + dist_worst)
    return np.nan_to_num(score, nan=0.0)


def clean_data(df: pd.DataFrame):
    df = df.fillna("Unknown")
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def main():
    print("📥 Model ve test verisi yükleniyor...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model bulunamadı: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)
    X = df.drop(columns=["readmit_30"], errors="ignore")

    # 🧹 Veri temizleme
    X_clean = clean_data(X)

    # 🧩 Model sütunlarını hizala
    model_features = model.feature_names_
    for col in model_features:
        if col not in X_clean.columns:
            X_clean[col] = 0
    X_clean = X_clean[model_features]

    # 🔮 Model tahmin olasılıkları
    y_pred_proba = model.predict_proba(X_clean)[:, 1]

    # 🔍 Özellik önemleri
    importance = model.get_feature_importance()
    features = X_clean.columns
    imp_df = pd.DataFrame({"feature": features, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    # En önemli 10 özelliği al
    top_features = imp_df.head(10)["feature"].tolist()

    # ❌ ICD sütunlarını çıkar
    exclude_features = ["diag_1_cat", "diag_2_cat", "diag_3_cat", "diag_1", "diag_2", "diag_3"]
    top_features = [f for f in top_features if f not in exclude_features]

    # ✅ Sayısal sütunlara indirgeme
    X_numeric = X_clean[top_features].select_dtypes(include=[np.number])
    top_features = X_numeric.columns.tolist()

    # Eğer hiç sayısal özellik kalmadıysa hata verme, tüm skorları 0 yap
    if len(top_features) == 0:
        print("⚠️  Sayısal özellik kalmadı, tüm skorlar 0 atandı.")
        scores = np.zeros(len(X_numeric))
        risk_level = ["Bilinmiyor"] * len(scores)
    else:
        matrix = X_numeric.to_numpy(dtype=float)
        scaler = MinMaxScaler()
        matrix = scaler.fit_transform(matrix)

        # Ağırlıkları hizala
        imp_df_filtered = imp_df[imp_df["feature"].isin(top_features)]
        weights = imp_df_filtered["importance"].values
        weights = weights / weights.sum()
        impacts = ["+"] * len(weights)

        # 🧮 TOPSIS skorlarını hesapla
        scores = topsis(matrix, weights, impacts)

        # Eğer skorlar tamamen 0 ise fallback
        if np.all(scores == 0) or np.isnan(scores).all():
            print("⚠️  TOPSIS skorları geçersiz (tüm değerler 0 veya NaN).")
            risk_level = ["Bilinmiyor"] * len(scores)
        else:
            risk_level = pd.cut(scores, bins=3,
                                labels=["Düşük Risk", "Orta Risk", "Yüksek Risk"],
                                duplicates="drop")

    # 📊 Sonuç tablosu
    output = pd.DataFrame({
        "PatientID": range(len(scores)),
        "TopsisScore": scores,
        "RiskLevel": risk_level,
        "PredictedProb": y_pred_proba
    })

    output.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ TOPSIS karar analizi tamamlandı.\n📁 Kaydedildi: {OUTPUT_PATH}")
    print(output.head())


if __name__ == "__main__":
    main()
