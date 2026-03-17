"""
Data Preprocessing Pipeline
Cleans, engineers features, encodes, and scales the dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

CATEGORICAL_OHE = ["ContractType", "SubscriptionPlan", "PaymentMethod", "MaritalStatus", "Gender"]
CATEGORICAL_BINARY = ["AutoPay", "DiscountReceived", "ContractRenewalReminder"]
DROP_COLS = ["CustomerID", "City"]


def load_raw(path="data/customers.csv"):
    return pd.read_csv(path)


def clean(df):
    df = df.drop_duplicates()
    # Age and Income range validation
    df = df[(df["Age"].between(18, 80)) & (df["Income"] > 0)]
    df["MonthlyCharges"] = df["MonthlyCharges"].clip(lower=0)
    df["TotalCharges"] = df["TotalCharges"].clip(lower=0)
    return df.reset_index(drop=True)


def feature_engineer(df):
    df["AverageMonthlySpend"] = df["TotalCharges"] / df["Tenure"].replace(0, 1)
    df["UsagePerDay"] = df["MonthlyUsageHours"] / 30
    df["SupportCallRate"] = df["CustomerSupportCalls"] / df["Tenure"].replace(0, 1)
    df["ComplaintRate"] = df["ServiceComplaints"] / df["Tenure"].replace(0, 1)
    df["EngagementScore"] = (
        df["LoginFrequency"] * 0.3
        + df["FeatureUsageScore"] * 0.4
        + df["AppRating"] * 10 * 0.3
    )
    return df


def encode(df):
    # Binary encoding
    for col in CATEGORICAL_BINARY:
        df[col] = (df[col] == "Yes").astype(int)

    # One-hot encoding
    df = pd.get_dummies(df, columns=CATEGORICAL_OHE, drop_first=False)

    # Drop non-informative cols
    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def preprocess(path="data/customers.csv", save_dir="data/"):
    os.makedirs(save_dir, exist_ok=True)

    df = load_raw(path)
    df = clean(df)
    df = feature_engineer(df)
    df = encode(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale (needed for SVM, KNN, Linear Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f"{save_dir}X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f"{save_dir}X_test_scaled.csv", index=False)
    X_train.to_csv(f"{save_dir}X_train.csv", index=False)
    X_test.to_csv(f"{save_dir}X_test.csv", index=False)
    y_train.to_csv(f"{save_dir}y_train.csv", index=False)
    y_test.to_csv(f"{save_dir}y_test.csv", index=False)
    joblib.dump(scaler, f"{save_dir}scaler.pkl")

    print(f"Preprocessing done. Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Churn rate (train): {y_train.mean():.1%} | Churn rate (test): {y_test.mean():.1%}")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    preprocess()
