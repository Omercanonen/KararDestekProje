import os
import joblib
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import xgboost as xgb

MODEL_PATH = "models/xgb/xgb_model.joblib"
DATA_PATH  = "data/processed/test.csv"
REPORT_DIR = "reports/figures"
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    print("📥 Model ve veri yükleniyor...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    y = df.pop("readmit_30")
    X = df.select_dtypes(include=["number", "bool"])

    # numpy formatına dönüştür
    X_np = X.values
    y_np = y.values

    # LIME açıklayıcıyı oluştur
    print("🧠 LIME açıklayıcı hazırlanıyor...")
    explainer = LimeTabularExplainer(
        training_data=X_np,
        feature_names=X.columns.tolist(),
        class_names=["No Readmit", "Readmit"],
        mode="classification"
    )

    # Test setinden bir örnek seç
    sample_idx = 42  # örnek hasta
    sample = X_np[sample_idx]
    print(f"🔍 Seçilen hasta indeksi: {sample_idx}")

    # Tahmin olasılığı için XGBoost fonksiyonu
    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=X.columns)
        d = xgb.DMatrix(x_df, feature_names=list(X.columns))
        preds = model.predict(d)
        return np.column_stack((1 - preds, preds))

    # LIME açıklamasını oluştur
    exp = explainer.explain_instance(sample, predict_fn, num_features=10)

    # Terminal çıktısı
    print("\n📊 En etkili 10 özellik:")
    for feat, val in exp.as_list():
        print(f"{feat:<30} {val:+.4f}")

    # Görselleştirme
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/lime_explanation_sample.png", dpi=300)
    plt.close()
    print("✅ LIME açıklaması kaydedildi:", f"{REPORT_DIR}/lime_explanation_sample.png")

if __name__ == "__main__":
    main()
