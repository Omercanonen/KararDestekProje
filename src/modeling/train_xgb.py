import os
import pandas as pd
import xgboost as xgb
import joblib
from src.utils.metrics import evaluate_binary

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/xgb"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    X_valid = pd.read_csv(f"{PROCESSED_DIR}/valid.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/test.csv")

    y_train = X_train.pop("readmit_30")
    y_valid = X_valid.pop("readmit_30")
    y_test = X_test.pop("readmit_30")

    # ❗ Sadece sayısal verileri tut
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns
    X_train = X_train[num_cols]
    X_valid = X_valid[num_cols]
    X_test = X_test[num_cols]

    print(f"Sadece sayısal sütunlar seçildi ({len(num_cols)} özellik)")
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_xgb_model(X_train, y_train, X_valid, y_valid):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "lambda": 1.0,
        "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),  # dengesiz veri ayarı
        "random_state": 42
    }

    watchlist = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    return model

def main():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    print("Eğitim verisi:", X_train.shape)

    # Modeli eğit
    model = train_xgb_model(X_train, y_train, X_valid, y_valid)
    joblib.dump(model, f"{MODEL_DIR}/xgb_model.joblib")
    print("✅ Model kaydedildi:", f"{MODEL_DIR}/xgb_model.joblib")

    # Tahmin ve metrikler
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    results = evaluate_binary(y_test, y_pred_proba)

    print("\n📊 Test Sonuçları:")
    for k, v in results.items():
        print(f"{k:<12}: {v:.4f}")

    # Rapor kaydı
    pd.DataFrame([results]).to_csv(f"{MODEL_DIR}/metrics.csv", index=False)
    print("\n📁 Metrikler kaydedildi:", f"{MODEL_DIR}/metrics.csv")

if __name__ == "__main__":
    main()
