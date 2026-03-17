"""
Customer Churn Dataset Generator
Generates a synthetic telecom/subscription dataset with 10,000 customers.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

N = 10_000

def generate_dataset(n=N):
    records = []

    for i in range(n):
        # --- Demographics ---
        age = int(np.random.normal(40, 12))
        age = max(18, min(80, age))
        gender = random.choice(["Male", "Female"])
        city = fake.city()
        income = int(np.random.normal(55000, 20000))
        income = max(15000, min(200000, income))
        marital_status = random.choices(
            ["Single", "Married", "Divorced"], weights=[0.35, 0.50, 0.15]
        )[0]
        dependents = int(np.clip(np.random.poisson(1.2), 0, 5))

        # --- Account Info ---
        tenure = int(np.random.exponential(24))
        tenure = max(1, min(72, tenure))
        contract_type = random.choices(
            ["Monthly", "Yearly", "Two-year"], weights=[0.50, 0.30, 0.20]
        )[0]
        plan = random.choices(
            ["Basic", "Standard", "Premium"], weights=[0.40, 0.35, 0.25]
        )[0]
        plan_price = {"Basic": 29, "Standard": 59, "Premium": 99}[plan]
        monthly_charges = round(plan_price + np.random.normal(0, 5), 2)
        monthly_charges = max(20, monthly_charges)
        total_charges = round(monthly_charges * tenure + np.random.normal(0, 50), 2)
        total_charges = max(0, total_charges)
        payment_method = random.choices(
            ["Credit Card", "Bank Transfer", "PayPal"], weights=[0.40, 0.35, 0.25]
        )[0]
        auto_pay = random.choices(["Yes", "No"], weights=[0.55, 0.45])[0]

        # --- Service Usage ---
        monthly_usage_hours = round(max(0, np.random.normal(40, 20)), 1)
        login_frequency = int(max(0, np.random.poisson(15)))
        feature_usage_score = round(max(0, min(100, np.random.normal(55, 20))), 1)
        support_calls = int(np.clip(np.random.poisson(1.5), 0, 15))
        complaints = int(np.clip(np.random.poisson(0.5), 0, 8))
        app_rating = round(max(1, min(5, np.random.normal(3.5, 0.8))), 1)

        # --- Behavior ---
        discount_received = random.choices(["Yes", "No"], weights=[0.30, 0.70])[0]
        last_interaction_days = int(np.clip(np.random.exponential(30), 1, 365))
        renewal_reminder = random.choices(["Yes", "No"], weights=[0.50, 0.50])[0]
        upgrade_attempts = int(np.clip(np.random.poisson(0.8), 0, 5))

        # --- Churn Logic (realistic, not random) ---
        churn_score = 0.0
        if contract_type == "Monthly":       churn_score += 0.25
        if contract_type == "Yearly":        churn_score += 0.05
        if tenure < 6:                       churn_score += 0.20
        if support_calls >= 3:               churn_score += 0.15
        if complaints >= 2:                  churn_score += 0.20
        if app_rating <= 2:                  churn_score += 0.15
        if last_interaction_days > 60:       churn_score += 0.15
        if monthly_usage_hours < 10:         churn_score += 0.10
        if login_frequency < 5:              churn_score += 0.10
        if discount_received == "Yes":       churn_score -= 0.10
        if auto_pay == "Yes":                churn_score -= 0.08
        if renewal_reminder == "Yes":        churn_score -= 0.05
        if feature_usage_score > 70:         churn_score -= 0.10
        if tenure > 36:                      churn_score -= 0.15
        churn_score = max(0.05, min(0.95, churn_score + np.random.normal(0, 0.05)))
        churn = int(np.random.random() < churn_score)

        records.append({
            "CustomerID": f"C{i+1:05d}",
            "Age": age,
            "Gender": gender,
            "City": city,
            "Income": income,
            "MaritalStatus": marital_status,
            "Dependents": dependents,
            "Tenure": tenure,
            "ContractType": contract_type,
            "SubscriptionPlan": plan,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "PaymentMethod": payment_method,
            "AutoPay": auto_pay,
            "MonthlyUsageHours": monthly_usage_hours,
            "LoginFrequency": login_frequency,
            "FeatureUsageScore": feature_usage_score,
            "CustomerSupportCalls": support_calls,
            "ServiceComplaints": complaints,
            "AppRating": app_rating,
            "DiscountReceived": discount_received,
            "LastInteractionDays": last_interaction_days,
            "ContractRenewalReminder": renewal_reminder,
            "UpgradeAttempts": upgrade_attempts,
            "Churn": churn,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset()
    df.to_csv("data/customers.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, {len(df.columns)} columns")
    print(f"Churn rate: {df['Churn'].mean():.1%}")
