"""
Model Training
Trains: Linear Regression (baseline), Logistic Regression, Decision Tree,
        SVM, KNN, and K-Means clustering.
Saves all models + comparison results.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

MODEL_DIR = "models/"


def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    X_train_s = pd.read_csv("data/X_train_scaled.csv")
    X_test_s = pd.read_csv("data/X_test_scaled.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return X_train, X_test, X_train_s, X_test_s, y_train, y_test


def evaluate(model, X_test, y_test, model_name, uses_proba=True, is_linear_reg=False):
    if is_linear_reg:
        y_prob = model.predict(X_test).clip(0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
    elif uses_proba:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred  # SVM without proba

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc else None,
        "confusion_matrix": cm,
    }


def train_all():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, X_test, X_train_s, X_test_s, y_train, y_test = load_data()

    results = []

    # 1. Linear Regression (baseline)
    print("Training Linear Regression (baseline)...")
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    joblib.dump(lr, f"{MODEL_DIR}linear_regression.pkl")
    results.append(evaluate(lr, X_test_s, y_test, "Linear Regression", is_linear_reg=True))

    # 2. Logistic Regression
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_s, y_train)
    joblib.dump(log_reg, f"{MODEL_DIR}logistic_regression.pkl")
    results.append(evaluate(log_reg, X_test_s, y_test, "Logistic Regression"))

    # 3. Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42)
    dt.fit(X_train, y_train)
    joblib.dump(dt, f"{MODEL_DIR}decision_tree.pkl")
    results.append(evaluate(dt, X_test, y_test, "Decision Tree"))

    # 4. SVM
    print("Training SVM (this may take a moment)...")
    svm = SVC(kernel="rbf", probability=True, random_state=42, C=1.0)
    svm.fit(X_train_s, y_train)
    joblib.dump(svm, f"{MODEL_DIR}svm.pkl")
    results.append(evaluate(svm, X_test_s, y_test, "SVM"))

    # 5. KNN
    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_s, y_train)
    joblib.dump(knn, f"{MODEL_DIR}knn.pkl")
    results.append(evaluate(knn, X_test_s, y_test, "KNN"))

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{MODEL_DIR}model_comparison.csv", index=False)
    print("\n=== Model Comparison ===")
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))

    # 6. K-Means clustering
    print("\nTraining K-Means (4 segments)...")
    scaler = joblib.load("data/scaler.pkl")
    X_all_s = scaler.transform(pd.concat([X_train, X_test]))
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_all_s)
    joblib.dump(kmeans, f"{MODEL_DIR}kmeans.pkl")

    # Label clusters
    X_all = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_all = pd.concat([y_train, y_test]).reset_index(drop=True)
    X_all["Cluster"] = kmeans.labels_
    X_all["Churn"] = y_all.values

    cluster_summary = X_all.groupby("Cluster").agg(
        CustomerCount=("Churn", "count"),
        ChurnRate=("Churn", "mean"),
        AvgTenure=("Tenure", "mean"),
        AvgMonthlyCharges=("MonthlyCharges", "mean"),
        AvgSupportCalls=("CustomerSupportCalls", "mean"),
        AvgAppRating=("AppRating", "mean"),
    ).round(3)

    # Auto-label clusters based on churn rate
    churn_ranks = cluster_summary["ChurnRate"].rank()
    labels = {}
    for idx in cluster_summary.index:
        rate = cluster_summary.loc[idx, "ChurnRate"]
        tenure = cluster_summary.loc[idx, "AvgTenure"]
        if rate <= cluster_summary["ChurnRate"].quantile(0.25):
            labels[idx] = "Loyal Customers"
        elif rate <= cluster_summary["ChurnRate"].quantile(0.50):
            labels[idx] = "High-Value Customers"
        elif rate <= cluster_summary["ChurnRate"].quantile(0.75):
            labels[idx] = "At-Risk Customers"
        else:
            labels[idx] = "Churn-Risk Customers"

    cluster_summary["Label"] = pd.Series(labels)
    cluster_summary.to_csv(f"{MODEL_DIR}cluster_summary.csv")
    print("\nCluster Summary:")
    print(cluster_summary)

    print("\nAll models saved to models/")
    return results_df, cluster_summary


if __name__ == "__main__":
    train_all()
