from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score
)

def evaluate_binary(y_true, y_pred_proba, threshold=0.5):
    # Tahmin olasılığı → sınıfa dönüştür
    y_pred = (y_pred_proba >= threshold).astype(int)

    # ROC ve PR-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(rec, prec)

    # Klasik metrikler
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "f1": f1
    }
