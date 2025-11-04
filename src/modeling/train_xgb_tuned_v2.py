import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, roc_curve
)

DATA_PATH = "data/processed/train.csv"
VALID_PATH = "data/processed/valid.csv"
MODEL_DIR = "models/xgb_tuned_v2"
os.makedirs(MODEL_DIR, exist_ok=True)


def feature_engineering(df):

    df["hospital_visits_total"] = (
        df["number_inpatient"] + df["number_emergency"] + df["number_outpatient"]
    )
    df["long_stay_flag"] = (df["time_in_hospital"] > 7).astype(int)
    df["medication_density"] = df["num_medications"] / (df["num_lab_procedures"] + 1)
    return df


def main():
    print("📥 Veri yükleniyor...")
    train_df = pd.read_csv(DATA_PATH)
    valid_df = pd.read_csv(VALID_PATH)

    X_train = train_df.drop(columns=["readmit_30"]).select_dtypes(include=["number", "bool"])
    y_train = train_df["readmit_30"]
    X_valid = valid_df.drop(columns=["readmit_30"]).select_dtypes(include=["number", "bool"])
    y_valid = valid_df["readmit_30"]

    # 🧩 Feature Engineering
    X_train = feature_engineering(X_train)
    X_valid = feature_engineering(X_valid)

    # ⚖️ ADASYN ile dengesizlik gideriliyor
    print("⚖️  ADASYN ile sınıf dengesi sağlanıyor...")
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X_train, y_train)
    print(f"Önce: {y_train.value_counts().to_dict()} → Sonra: {pd.Series(y_res).value_counts().to_dict()}")

    # ⚙️ Sınıf ağırlığı hesapla
    scale = len(y_res[y_res == 0]) / len(y_res[y_res == 1])

    # 🔍 Geniş parametre arama alanı
    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08],
        "n_estimators": [800, 1200],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "gamma": [0, 0.3, 0.5],
        "min_child_weight": [1, 3, 5],
    }

    print("🔍 Geniş GridSearch başlatılıyor (birkaç dakika sürebilir)...")
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_res, y_res)
    best_model = grid.best_estimator_

    print(f"✅ En iyi parametreler: {grid.best_params_}")

    # 📊 Tahminler
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # 🔧 En iyi threshold'u otomatik bul
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)
    best_idx = (tpr - fpr).argmax()
    best_threshold = thresholds[best_idx]
    print(f"🔧 En iyi olasılık eşiği: {best_threshold:.2f}")

    y_pred = (y_pred_proba > best_threshold).astype(int)

    # 📈 Performans metrikleri
    metrics = {
        "roc_auc": roc_auc_score(y_valid, y_pred_proba),
        "accuracy": accuracy_score(y_valid, y_pred),
        "f1": f1_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred),
        "recall": recall_score(y_valid, y_pred)
    }

    print("\n📊 Yeni Model Performansı:")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")

    # 💾 Model kaydet
    joblib.dump(best_model, os.path.join(MODEL_DIR, "xgb_tuned_model.joblib"))
    pd.DataFrame([metrics]).to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)

    print(f"\n💾 Model kaydedildi: {MODEL_DIR}/xgb_tuned_model.joblib")


if __name__ == "__main__":
    main()
