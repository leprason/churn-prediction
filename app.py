"""
Customer Churn Prediction – Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/customers.csv")
    X_test = pd.read_csv("data/X_test.csv")
    X_test_s = pd.read_csv("data/X_test_scaled.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return df_raw, X_test, X_test_s, y_test

@st.cache_resource
def load_models():
    models = {}
    for name, fname, scaled in [
        ("Linear Regression",  "linear_regression.pkl",  True),
        ("Logistic Regression","logistic_regression.pkl", True),
        ("Decision Tree",      "decision_tree.pkl",       False),
        ("SVM",                "svm.pkl",                 True),
        ("KNN",                "knn.pkl",                 True),
    ]:
        path = f"models/{fname}"
        if os.path.exists(path):
            models[name] = (joblib.load(path), scaled)
    kmeans   = joblib.load("models/kmeans.pkl")   if os.path.exists("models/kmeans.pkl")   else None
    scaler   = joblib.load("data/scaler.pkl")     if os.path.exists("data/scaler.pkl")     else None
    return models, kmeans, scaler

@st.cache_data
def load_results():
    comp = pd.read_csv("models/model_comparison.csv") if os.path.exists("models/model_comparison.csv") else None
    clus = pd.read_csv("models/cluster_summary.csv", index_col=0) if os.path.exists("models/cluster_summary.csv") else None
    return comp, clus

# ── Check if trained ──────────────────────────────────────────────────────────
data_ready  = os.path.exists("data/customers.csv")
model_ready = os.path.exists("models/model_comparison.csv")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔄 Pipeline Control")

if st.sidebar.button("1️⃣  Generate Dataset", use_container_width=True):
    with st.spinner("Generating 10,000 customers..."):
        os.makedirs("data", exist_ok=True)
        import generate_dataset
        df = generate_dataset.generate_dataset()
        df.to_csv("data/customers.csv", index=False)
        st.sidebar.success(f"Dataset ready! Churn rate: {df['Churn'].mean():.1%}")
        st.cache_data.clear()

if st.sidebar.button("2️⃣  Preprocess Data", use_container_width=True, disabled=not data_ready):
    with st.spinner("Preprocessing..."):
        import preprocessing
        preprocessing.preprocess()
        st.sidebar.success("Preprocessing complete!")
        st.cache_data.clear()

if st.sidebar.button("3️⃣  Train All Models", use_container_width=True, disabled=not os.path.exists("data/X_train.csv")):
    with st.spinner("Training models (SVM may take ~60s)..."):
        import train_models
        train_models.train_all()
        st.sidebar.success("All models trained!")
        st.cache_resource.clear()
        st.cache_data.clear()

st.sidebar.divider()
page = st.sidebar.radio("📋 Navigate", [
    "🏠 Overview",
    "🔍 EDA",
    "🤖 Model Comparison",
    "📊 Confusion Matrices",
    "👥 Customer Segments",
    "🎯 Predict Single Customer",
])

# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("📊 Customer Churn Prediction System")
    st.markdown("AI-powered churn prediction for telecom/subscription services.")

    if not data_ready:
        st.info("👈 Click **Generate Dataset** in the sidebar to get started.")
    else:
        df_raw, *_ = load_data()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(df_raw):,}")
        c2.metric("Churn Rate", f"{df_raw['Churn'].mean():.1%}")
        c3.metric("Features", "24")
        c4.metric("Models Trained", "5 + K-Means")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Sample")
            st.dataframe(df_raw.head(8), use_container_width=True)
        with col2:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(figsize=(4, 3))
            counts = df_raw["Churn"].value_counts()
            ax.pie(counts, labels=["Stays", "Churns"], autopct="%1.1f%%",
                   colors=["#4CAF50", "#F44336"], startangle=90)
            ax.set_title("Overall Churn Split")
            st.pyplot(fig)
            plt.close()

# ── Page: EDA ────────────────────────────────────────────────────────────────
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    if not data_ready:
        st.warning("Generate dataset first.")
    else:
        df_raw, *_ = load_data()

        tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Churn Drivers"])

        with tab1:
            col = st.selectbox("Select numeric feature", ["Age", "Tenure", "MonthlyCharges",
                "MonthlyUsageHours", "LoginFrequency", "AppRating",
                "CustomerSupportCalls", "ServiceComplaints"])
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            df_raw[col].hist(bins=30, ax=axes[0], color="#2196F3", edgecolor="white")
            axes[0].set_title(f"{col} Distribution")
            df_raw.boxplot(column=col, by="Churn", ax=axes[1])
            axes[1].set_title(f"{col} by Churn")
            axes[1].set_xlabel("Churn (0=Stay, 1=Churn)")
            plt.suptitle("")
            st.pyplot(fig)
            plt.close()

        with tab2:
            numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
            corr = df_raw[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                        center=0, linewidths=0.5, ax=ax)
            ax.set_title("Feature Correlation Matrix")
            st.pyplot(fig)
            plt.close()

        with tab3:
            st.subheader("Churn Rate by Key Categorical Features")
            for feat in ["ContractType", "SubscriptionPlan", "AutoPay"]:
                fig, ax = plt.subplots(figsize=(6, 2.5))
                grouped = df_raw.groupby(feat)["Churn"].mean().sort_values(ascending=False)
                grouped.plot(kind="bar", ax=ax, color="#FF7043", edgecolor="white")
                ax.set_title(f"Churn Rate by {feat}")
                ax.set_ylabel("Churn Rate")
                ax.set_ylim(0, 1)
                ax.tick_params(axis="x", rotation=0)
                for bar in ax.patches:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.01,
                            f"{bar.get_height():.0%}", ha="center", fontsize=9)
                st.pyplot(fig)
                plt.close()

# ── Page: Model Comparison ────────────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    if not model_ready:
        st.warning("Train models first.")
    else:
        comp, _ = load_results()
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        st.subheader("📋 Results Table")
        styled = comp[["model"] + metrics].copy()
        st.dataframe(
            styled.style.background_gradient(subset=metrics, cmap="YlGn"),
            use_container_width=True, hide_index=True
        )

        st.subheader("📈 Metric Radar")
        selected_metric = st.selectbox("Compare by metric", metrics)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
        bars = ax.barh(comp["model"], comp[selected_metric], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel(selected_metric.replace("_", " ").title())
        ax.set_title(f"Model Comparison – {selected_metric.replace('_',' ').title()}")
        for bar, val in zip(bars, comp[selected_metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
        st.pyplot(fig)
        plt.close()

        best = comp.loc[comp["f1_score"].idxmax(), "model"]
        best_f1 = comp["f1_score"].max()
        st.success(f"🏆 Best model by F1 Score: **{best}** ({best_f1:.3f})")

# ── Page: Confusion Matrices ──────────────────────────────────────────────────
elif page == "📊 Confusion Matrices":
    st.title("📊 Confusion Matrices")
    if not model_ready:
        st.warning("Train models first.")
    else:
        comp, _ = load_results()
        _, X_test, X_test_s, y_test = load_data()
        models, _, _ = load_models()

        cols = st.columns(3)
        for idx, (name, (model, scaled)) in enumerate(models.items()):
            X_input = X_test_s if scaled else X_test
            try:
                if name == "Linear Regression":
                    y_pred = (model.predict(X_input) >= 0.5).astype(int)
                else:
                    y_pred = model.predict(X_input)
                # Always recompute CM directly from predictions
                from sklearn.metrics import confusion_matrix as sk_cm
                cm = sk_cm(y_test, y_pred)
            except Exception as e:
                st.warning(f"{name}: {e}")
                continue

            with cols[idx % 3]:
                fig, ax = plt.subplots(figsize=(3.5, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Stay", "Churn"],
                            yticklabels=["Stay", "Churn"], ax=ax)
                ax.set_title(name, fontsize=10)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                plt.close()

# ── Page: Customer Segments ───────────────────────────────────────────────────
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments (K-Means)")
    if not model_ready:
        st.warning("Train models first.")
    else:
        _, clus = load_results()
        if clus is not None:
            st.subheader("Cluster Summary")
            st.dataframe(clus, use_container_width=True)

            segment_colors = {"Loyal Customers": "#4CAF50",
                              "High-Value Customers": "#2196F3",
                              "At-Risk Customers": "#FF9800",
                              "Churn-Risk Customers": "#F44336"}

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                labels = clus["Label"].tolist()
                sizes  = clus["CustomerCount"].tolist()
                colors = [segment_colors.get(l, "#9E9E9E") for l in labels]
                ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
                ax.set_title("Segment Distribution")
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                x = range(len(clus))
                ax.bar(x, clus["ChurnRate"], color=colors)
                ax.set_xticks(list(x))
                ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
                ax.set_ylabel("Churn Rate")
                ax.set_title("Churn Rate per Segment")
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                plt.close()

# ── Page: Predict Single Customer ─────────────────────────────────────────────
elif page == "🎯 Predict Single Customer":
    st.title("🎯 Predict Individual Customer Churn")
    if not model_ready:
        st.warning("Train models first.")
    else:
        models, _, scaler = load_models()

        st.subheader("Enter Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 80, 35)
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 59)
            contract_type = st.selectbox("Contract Type", ["Monthly", "Yearly", "Two-year"])

        with col2:
            plan = st.selectbox("Subscription Plan", ["Basic", "Standard", "Premium"])
            support_calls = st.slider("Support Calls", 0, 15, 2)
            complaints = st.slider("Complaints", 0, 8, 0)
            app_rating = st.slider("App Rating", 1.0, 5.0, 3.5, step=0.1)

        with col3:
            usage_hours = st.slider("Monthly Usage Hours", 0, 150, 40)
            login_freq = st.slider("Login Frequency", 0, 60, 15)
            last_interaction = st.slider("Days Since Last Interaction", 1, 365, 30)
            auto_pay = st.selectbox("AutoPay", ["Yes", "No"])

        selected_model = st.selectbox("Choose Model", list(models.keys()))

        if st.button("🔮 Predict Churn", use_container_width=True):
            # Build feature vector to match training columns
            X_test_ref = pd.read_csv("data/X_test.csv")
            feature_cols = X_test_ref.columns.tolist()

            total_charges = monthly_charges * tenure
            feature_usage = 55.0
            discount = 0
            renewal_reminder = 1
            upgrade_attempts = 0
            income = 55000
            dependents = 1

            # Engineer features
            avg_monthly_spend = total_charges / max(tenure, 1)
            usage_per_day = usage_hours / 30
            support_call_rate = support_calls / max(tenure, 1)
            complaint_rate = complaints / max(tenure, 1)
            engagement_score = login_freq * 0.3 + feature_usage * 0.4 + app_rating * 10 * 0.3

            row = {col: 0 for col in feature_cols}
            # Fill numeric
            for k, v in {
                "Age": age, "Income": income, "Dependents": dependents,
                "Tenure": tenure, "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges, "MonthlyUsageHours": usage_hours,
                "LoginFrequency": login_freq, "FeatureUsageScore": feature_usage,
                "CustomerSupportCalls": support_calls, "ServiceComplaints": complaints,
                "AppRating": app_rating, "LastInteractionDays": last_interaction,
                "UpgradeAttempts": upgrade_attempts,
                "AutoPay": 1 if auto_pay == "Yes" else 0,
                "DiscountReceived": discount,
                "ContractRenewalReminder": renewal_reminder,
                "AverageMonthlySpend": avg_monthly_spend,
                "UsagePerDay": usage_per_day,
                "SupportCallRate": support_call_rate,
                "ComplaintRate": complaint_rate,
                "EngagementScore": engagement_score,
            }.items():
                if k in row:
                    row[k] = v

            # One-hot
            for col in feature_cols:
                if col == f"ContractType_{contract_type}":    row[col] = 1
                if col == f"SubscriptionPlan_{plan}":         row[col] = 1
                if col == "Gender_Male":                      row[col] = 1
                if col == "MaritalStatus_Single":             row[col] = 1
                if col == "PaymentMethod_Credit Card":        row[col] = 1

            X_input = pd.DataFrame([row])[feature_cols]
            model_obj, uses_scaled = models[selected_model]
            X_final = scaler.transform(X_input) if uses_scaled else X_input

            if selected_model == "Linear Regression":
                prob = float(np.clip(model_obj.predict(X_final)[0], 0, 1))
            else:
                prob = float(model_obj.predict_proba(X_final)[0][1])

            churn = prob >= 0.5
            risk = "🔴 HIGH" if prob > 0.7 else ("🟡 MEDIUM" if prob > 0.4 else "🟢 LOW")

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability", f"{prob:.1%}")
            r2.metric("Risk Level", risk)
            r3.metric("Prediction", "Will Churn" if churn else "Will Stay")

            st.subheader("💡 Suggested Actions")
            actions = []
            if contract_type == "Monthly":  actions.append("Offer a discounted annual contract")
            if support_calls >= 3:          actions.append("Escalate to retention team")
            if complaints >= 2:             actions.append("Proactive service quality review")
            if app_rating <= 2:             actions.append("Send app feedback survey + credits")
            if last_interaction > 60:       actions.append("Trigger re-engagement campaign")
            if not actions:                 actions.append("Customer is healthy — monitor regularly")
            for a in actions:
                st.markdown(f"• {a}")
