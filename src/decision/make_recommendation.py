import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

INPUT_PATH = "data/processed/test.csv"
MODEL_PATH = "models/xgb/xgb_model.joblib"
OUTPUT_DIR = "reports/recommendations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def classify_risk(prob):
    if prob < 0.34:
        return "Düşük Risk"
    elif prob < 0.67:
        return "Orta Risk"
    else:
        return "Yüksek Risk"

def generate_recommendation(risk_level):
    if risk_level == "Düşük Risk":
        return "İlaç tedavisi yeterli, rutin kontroller devam etsin."
    elif risk_level == "Orta Risk":
        return "Hemşire gözetiminde haftalık takip önerilir."
    else:
        return "Doktor gözetiminde sıkı izlem ve ilaç düzenlemesi yapılmalıdır."

def main():
    print("📥 Veri ve model yükleniyor...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(INPUT_PATH)
    y_true = df["readmit_30"]
    X = df.drop(columns=["readmit_30"]).select_dtypes(include=["number", "bool"])

    dtest = xgb.DMatrix(X, feature_names=list(X.columns))
    y_pred_proba = model.predict(dtest)

    df_result = pd.DataFrame({
        "PatientID": np.arange(len(X)),
        "TrueLabel": y_true,
        "PredictedProb": y_pred_proba
    })

    # Risk sınıfı ve öneri oluştur
    df_result["RiskLevel"] = df_result["PredictedProb"].apply(classify_risk)
    df_result["Recommendation"] = df_result["RiskLevel"].apply(generate_recommendation)

    out_csv = os.path.join(OUTPUT_DIR, "recommendations.csv")
    df_result.to_csv(out_csv, index=False)

    print(f"✅ {len(df_result)} hasta için öneriler oluşturuldu.")
    print(f"📁 Kaydedilen dosya: {out_csv}")

    print("\n📊 Örnek kayıtlar:")
    print(df_result.head(5))

if __name__ == "__main__":
    main()
