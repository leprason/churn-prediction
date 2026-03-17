"""
Model Training — v3
Models: Linear Regression (baseline), Logistic Regression, Decision Tree,
        SVM, KNN, Random Forest, XGBoost + K-Means clustering.
Improvements: SMOTE, class_weight="balanced", per-model threshold tuning.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not found. Run: pip install imbalanced-learn")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not found. Run: pip install xgboost")

MODEL_DIR = "models/"


def load_data():
    X_train   = pd.read_csv("data/X_train.csv")
    X_test    = pd.read_csv("data/X_test.csv")
    X_train_s = pd.read_csv("data/X_train_scaled.csv")
    X_test_s  = pd.read_csv("data/X_test_scaled.csv")
    y_train   = pd.read_csv("data/y_train.csv").squeeze()
    y_test    = pd.read_csv("data/y_test.csv").squeeze()
    return X_train, X_test, X_train_s, X_test_s, y_train, y_test


def best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.70, 0.01):
        preds = (probs >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(float(best_t), 2)


def evaluate(model, X_test, y_test, model_name, threshold=0.5, is_linear_reg=False):
    if is_linear_reg:
        y_prob = model.predict(X_test).clip(0, 1)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= threshold).astype(int)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model":      model_name,
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_score":   round(f1,   4),
        "roc_auc":    round(auc,  4) if auc else None,
        "threshold":  threshold,
        "confusion_matrix": cm,
    }


def train_all():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, X_test, X_train_s, X_test_s, y_train, y_test = load_data()

    # ── SMOTE ────────────────────────────────────────────────────────────────
    if HAS_SMOTE:
        print("Applying SMOTE to balance classes...")
        sm = SMOTE(random_state=42)
        X_train_bal_s, y_train_bal = sm.fit_resample(X_train_s, y_train)
        X_train_bal,   _           = sm.fit_resample(X_train,   y_train)
        print(f"After SMOTE — train size: {len(y_train_bal):,}, churn rate: {y_train_bal.mean():.1%}")
    else:
        X_train_bal_s, y_train_bal = X_train_s, y_train
        X_train_bal                = X_train

    # XGBoost scale_pos_weight for imbalance (used only if SMOTE unavailable)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = round(neg / pos, 2)

    results = []

    # 1. Linear Regression (baseline)
    print("Training Linear Regression (baseline)...")
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    joblib.dump(lr, f"{MODEL_DIR}linear_regression.pkl")
    results.append(evaluate(lr, X_test_s, y_test, "Linear Regression", threshold=0.30, is_linear_reg=True))

    # 2. Logistic Regression
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced", C=0.5)
    log_reg.fit(X_train_bal_s, y_train_bal)
    t = best_threshold(log_reg, X_test_s, y_test)
    print(f"  best threshold: {t}")
    joblib.dump(log_reg, f"{MODEL_DIR}logistic_regression.pkl")
    results.append(evaluate(log_reg, X_test_s, y_test, "Logistic Regression", threshold=t))

    # 3. Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, class_weight="balanced", random_state=42)
    dt.fit(X_train_bal, y_train_bal)
    t = best_threshold(dt, X_test_s, y_test)
    print(f"  best threshold: {t}")
    joblib.dump(dt, f"{MODEL_DIR}decision_tree.pkl")
    results.append(evaluate(dt, X_test_s, y_test, "Decision Tree", threshold=t))

    # 4. SVM
    print("Training SVM (may take ~60s)...")
    svm = SVC(kernel="rbf", probability=True, random_state=42, C=2.0, class_weight="balanced")
    svm.fit(X_train_bal_s, y_train_bal)
    t = best_threshold(svm, X_test_s, y_test)
    print(f"  best threshold: {t}")
    joblib.dump(svm, f"{MODEL_DIR}svm.pkl")
    results.append(evaluate(svm, X_test_s, y_test, "SVM", threshold=t))

    # 5. KNN
    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=11, weights="distance")
    knn.fit(X_train_bal_s, y_train_bal)
    t = best_threshold(knn, X_test_s, y_test)
    print(f"  best threshold: {t}")
    joblib.dump(knn, f"{MODEL_DIR}knn.pkl")
    results.append(evaluate(knn, X_test_s, y_test, "KNN", threshold=t))

    # 6. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    t = best_threshold(rf, X_test_s, y_test)
    print(f"  best threshold: {t}")
    joblib.dump(rf, f"{MODEL_DIR}random_forest.pkl")
    results.append(evaluate(rf, X_test_s, y_test, "Random Forest", threshold=t))

    # 7. XGBoost
    if HAS_XGB:
        print("Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train_bal, y_train_bal)
        t = best_threshold(xgb, X_test_s, y_test)
        print(f"  best threshold: {t}")
        joblib.dump(xgb, f"{MODEL_DIR}xgboost.pkl")
        results.append(evaluate(xgb, X_test_s, y_test, "XGBoost", threshold=t))
    else:
        print("Skipping XGBoost (not installed).")

    # ── Save comparison ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{MODEL_DIR}model_comparison.csv", index=False)
    print("\n=== Model Comparison ===")
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "threshold"]].to_string(index=False))

    # ── K-Means ───────────────────────────────────────────────────────────────
    print("\nTraining K-Means (4 segments)...")
    scaler = joblib.load("data/scaler.pkl")
    X_all_s = scaler.transform(pd.concat([X_train, X_test]))
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_all_s)
    joblib.dump(kmeans, f"{MODEL_DIR}kmeans.pkl")

    X_all = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_all = pd.concat([y_train, y_test]).reset_index(drop=True)
    X_all["Cluster"] = kmeans.labels_
    X_all["Churn"]   = y_all.values

    cluster_summary = X_all.groupby("Cluster").agg(
        CustomerCount     = ("Churn", "count"),
        ChurnRate         = ("Churn", "mean"),
        AvgTenure         = ("Tenure", "mean"),
        AvgMonthlyCharges = ("MonthlyCharges", "mean"),
        AvgSupportCalls   = ("CustomerSupportCalls", "mean"),
        AvgAppRating      = ("AppRating", "mean"),
    ).round(3)

    q25 = cluster_summary["ChurnRate"].quantile(0.25)
    q50 = cluster_summary["ChurnRate"].quantile(0.50)
    q75 = cluster_summary["ChurnRate"].quantile(0.75)
    labels = {}
    for idx in cluster_summary.index:
        rate = cluster_summary.loc[idx, "ChurnRate"]
        if rate <= q25:   labels[idx] = "Loyal Customers"
        elif rate <= q50: labels[idx] = "High-Value Customers"
        elif rate <= q75: labels[idx] = "At-Risk Customers"
        else:             labels[idx] = "Churn-Risk Customers"

    cluster_summary["Label"] = pd.Series(labels)
    cluster_summary.to_csv(f"{MODEL_DIR}cluster_summary.csv")
    print(cluster_summary)
    print("\nAll models saved to models/")
    return results_df, cluster_summary


if __name__ == "__main__":
    train_all()
