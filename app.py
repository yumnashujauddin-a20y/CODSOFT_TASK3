# ============================================================
# CHURNAI - CUSTOMER CHURN INTELLIGENCE PLATFORM
# Complete upgraded Streamlit application
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnAI | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
MODEL_PATH = BASE_DIR / "customer_churn_model.joblib"
METADATA_PATH = BASE_DIR / "customer_churn_model_metadata.json"


# ============================================================
# MODEL FEATURES
# IMPORTANT: These must match the features used during training.
# ============================================================

MODEL_FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

TARGET = "Exited"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 5% 0%, rgba(99,102,241,.18), transparent 27%),
        radial-gradient(circle at 95% 5%, rgba(14,165,233,.12), transparent 27%),
        #080d1a;
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

#MainMenu, header, footer {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid rgba(255,255,255,.08);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.hero {
    padding: 34px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(79,70,229,.30), rgba(14,165,233,.08)),
        rgba(15,23,42,.72);
    border: 1px solid rgba(255,255,255,.09);
    box-shadow: 0 25px 70px rgba(0,0,0,.28);
    margin-bottom: 25px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(99,102,241,.18);
    border: 1px solid rgba(129,140,248,.28);
    color: #c7d2fe;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.1px;
}

.hero-title {
    font-size: 46px;
    font-weight: 900;
    letter-spacing: -1.8px;
    color: white;
    line-height: 1.08;
    margin-top: 10px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 9px;
    max-width: 900px;
    line-height: 1.65;
}

.section-title {
    font-size: 23px;
    font-weight: 850;
    color: #f8fafc;
    margin-top: 26px;
    margin-bottom: 14px;
}

.kpi-card {
    background: linear-gradient(145deg, rgba(17,24,39,.96), rgba(15,23,42,.82));
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px;
    padding: 21px;
    min-height: 130px;
    box-shadow: 0 14px 40px rgba(0,0,0,.20);
}

.kpi-label {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .9px;
    font-weight: 800;
}

.kpi-value {
    color: white;
    font-size: 30px;
    font-weight: 900;
    margin-top: 7px;
}

.kpi-description {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.glass-card {
    background: rgba(15,23,42,.78);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px;
    padding: 23px;
    box-shadow: 0 15px 45px rgba(0,0,0,.18);
}

.info-box {
    padding: 18px;
    border-radius: 15px;
    background: rgba(30,41,59,.60);
    border: 1px solid rgba(255,255,255,.06);
    color: #94a3b8;
    line-height: 1.6;
}

.risk-high {
    color: #fb7185;
    font-size: 30px;
    font-weight: 900;
}

.risk-medium {
    color: #fbbf24;
    font-size: 30px;
    font-weight: 900;
}

.risk-low {
    color: #34d399;
    font-size: 30px;
    font-weight: 900;
}

.stButton > button {
    border-radius: 13px;
    min-height: 48px;
    font-weight: 800;
    border: 1px solid rgba(255,255,255,.08);
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    box-shadow: 0 9px 28px rgba(79,70,229,.25);
}

.stButton > button:hover {
    border-color: #818cf8;
    transform: translateY(-1px);
}

div[data-testid="stMetric"] {
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 17px;
    padding: 14px;
}

.footer {
    text-align: center;
    color: #475569;
    font-size: 12px;
    padding-top: 35px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def make_risk(probability):
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def risk_class(risk):
    return {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }.get(risk, "risk-low")


def create_gauge(value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value) * 100,
            number={
                "suffix": "%",
                "font": {"size": 40, "color": "white"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#64748b",
                },
                "bar": {"color": "#6366f1"},
                "bgcolor": "#111827",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#12372a"},
                    {"range": [30, 60], "color": "#3b3213"},
                    {"range": [60, 100], "color": "#421c27"},
                ],
            },
        )
    )
    fig.update_layout(
        height=285,
        margin=dict(l=20, r=20, t=25, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def validate_columns(frame, columns=MODEL_FEATURES):
    missing = [c for c in columns if c not in frame.columns]
    return missing


def prepare_model_input(frame):
    """Return only the exact columns expected by the saved model."""
    missing = validate_columns(frame)
    if missing:
        raise ValueError(
            "The saved model expects these features, but they are missing: "
            + ", ".join(missing)
        )
    return frame[MODEL_FEATURES].copy()


def get_model_expected_features(model):
    """Try to inspect feature names saved by sklearn/pipeline."""
    for obj in [model, getattr(model, "named_steps", {}).get("preprocessor", None)]:
        if obj is not None:
            names = getattr(obj, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None


def chart_layout(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
    )
    return fig


# ============================================================
# LOAD DATA / MODEL
# ============================================================

df = load_data()
model = load_model()
metadata = load_metadata()

if df is None:
    st.error(f"Dataset not found:\n\n{DATA_PATH}")
    st.stop()

if model is None:
    st.error(
        "Trained model not found.\n\n"
        f"Expected file:\n{MODEL_PATH}\n\n"
        "Run your training script first."
    )
    st.stop()

if TARGET not in df.columns:
    st.error("The dataset does not contain the expected 'Exited' target column.")
    st.stop()

missing_dataset_features = validate_columns(df)
if missing_dataset_features:
    st.error(
        "The dataset is missing model features: "
        + ", ".join(missing_dataset_features)
    )
    st.stop()

# ============================================================
# DATA SUMMARY
# ============================================================

total_customers = len(df)
churned_customers = int(df[TARGET].sum())
active_customers = total_customers - churned_customers
churn_rate = churned_customers / total_customers if total_customers else 0

model_name = metadata.get("model", type(model).__name__)
roc_auc_meta = metadata.get("roc_auc", None)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:28px;font-weight:900;margin-bottom:2px;">
            📊 Churn<span style="color:#818cf8;">AI</span>
        </div>
        <div style="color:#64748b;font-size:12px;margin-bottom:20px;">
            Customer Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Churn Predictor",
            "👥 Customer Risk",
            "📈 Analytics",
            "🤖 Model Performance",
        ],
    )

    st.divider()

    st.caption("DATASET")
    st.write(f"👥 {total_customers:,} customers")
    st.write(f"🚨 {churned_customers:,} churned")
    st.write(f"📊 {churn_rate:.1%} churn rate")

    st.divider()
    st.caption("MODEL")
    st.write(f"🤖 {model_name}")

    st.divider()
    st.caption("ChurnAI v2.0 • ML Analytics")

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">CUSTOMER INTELLIGENCE PLATFORM</div>
        <div class="hero-title">
            Churn<span style="color:#818cf8;">AI</span>
        </div>
        <div class="hero-subtitle">
            Predict customer churn, identify high-risk customers, explore
            behavioral patterns, and turn machine-learning predictions into
            actionable retention decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)

    cards = [
        ("Total Customers", f"{total_customers:,}", "Customers in dataset"),
        ("Churned Customers", f"{churned_customers:,}", "Customers who exited"),
        ("Churn Rate", f"{churn_rate:.1%}", "Overall customer churn"),
        ("Model", model_name, "Active prediction model"),
    ]

    for col, (label, value, desc) in zip([k1, k2, k3, k4], cards):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value"
                         style="font-size:{22 if label == 'Model' else 30}px;">
                        {value}
                    </div>
                    <div class="kpi-description">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Churn Overview</div>',
        unsafe_allow_html=True,
    )

    chart1, chart2 = st.columns(2)

    with chart1:
        distribution = pd.DataFrame(
            {
                "Status": ["Active", "Churned"],
                "Customers": [active_customers, churned_customers],
            }
        )

        fig = px.pie(
            distribution,
            names="Status",
            values="Customers",
            hole=0.64,
            title="Customer Distribution",
        )
        fig.update_traces(textinfo="percent+label")
        chart_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        if "Geography" in df.columns:
            geo_df = (
                df.groupby("Geography")[TARGET]
                .mean()
                .reset_index()
            )
            geo_df["Churn Rate"] = geo_df[TARGET] * 100

            fig = px.bar(
                geo_df,
                x="Geography",
                y="Churn Rate",
                text="Churn Rate",
                title="Churn Rate by Geography",
            )
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )
            chart_layout(fig, 390)
            fig.update_yaxes(title="Churn Rate (%)")
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Customer Demographics</div>',
        unsafe_allow_html=True,
    )

    age_df = df.copy()
    age_df["Age Group"] = pd.cut(
        age_df["Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=["18–25", "26–35", "36–45", "46–55", "56+"],
    )

    age_churn = (
        age_df.groupby("Age Group", observed=True)[TARGET]
        .mean()
        .reset_index()
    )
    age_churn["Churn Rate"] = age_churn[TARGET] * 100

    fig = px.line(
        age_churn,
        x="Age Group",
        y="Churn Rate",
        markers=True,
        title="Churn Rate by Age Group",
    )
    chart_layout(fig, 360)
    fig.update_yaxes(title="Churn Rate (%)")
    fig.update_xaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CHURN PREDICTOR
# ============================================================

elif page == "🔮 Churn Predictor":

    st.markdown(
        '<div class="section-title">Individual Churn Predictor</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
        Enter a customer's profile below. ChurnAI will estimate the
        probability that this customer will churn and classify the customer
        into Low, Medium, or High risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 👤 Personal Profile")
        age = st.slider("Age", 18, 100, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"],
        )

    with c2:
        st.markdown("### 💳 Financial Profile")
        credit_score = st.slider("Credit Score", 300, 850, 650)
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0,
            step=5000.0,
        )
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=5000.0,
        )

    with c3:
        st.markdown("### 📈 Account Profile")
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        num_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4],
        )
        has_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"],
        )
        active_member = st.selectbox(
            "Active Member?",
            ["Yes", "No"],
        )

    st.write("")

    predict = st.button(
        "🔮 ANALYZE CUSTOMER",
        use_container_width=True,
    )

    if predict:
        customer = pd.DataFrame(
            {
                "CreditScore": [credit_score],
                "Geography": [geography],
                "Gender": [gender],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [num_products],
                "HasCrCard": [1 if has_card == "Yes" else 0],
                "IsActiveMember": [
                    1 if active_member == "Yes" else 0
                ],
                "EstimatedSalary": [estimated_salary],
            }
        )

        try:
            customer = prepare_model_input(customer)

            probability = float(
                model.predict_proba(customer)[0][1]
            )
            threshold = float(
                metadata.get("classification_threshold", 0.50)
            )
            prediction = int(probability >= threshold)
            risk = make_risk(probability)

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True,
            )

            result1, result2 = st.columns([1, 1])

            with result1:
                st.plotly_chart(
                    create_gauge(probability),
                    use_container_width=True,
                )

            with result2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-top:35px;">
                        <div style="
                            color:#94a3b8;
                            font-size:12px;
                            text-transform:uppercase;
                            font-weight:800;">
                            Risk Assessment
                        </div>

                        <div class="{risk_class(risk)}"
                             style="margin-top:9px;">
                            {risk}
                        </div>

                        <div style="color:#94a3b8;margin-top:9px;">
                            Churn probability
                        </div>

                        <div style="
                            color:white;
                            font-size:35px;
                            font-weight:900;">
                            {probability:.1%}
                        </div>

                        <div style="color:#94a3b8;margin-top:12px;">
                            Prediction:
                            <b style="color:white;">
                                {"Likely to Churn" if prediction else "Likely to Stay"}
                            </b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if risk == "High":
                st.error(
                    "🚨 HIGH RISK — Immediate retention action is recommended."
                )
                st.info(
                    "Recommended: personalized offer, proactive support, "
                    "loyalty benefit, or plan review."
                )
            elif risk == "Medium":
                st.warning(
                    "⚠️ MEDIUM RISK — Monitor this customer and consider "
                    "targeted engagement."
                )
            else:
                st.success(
                    "✅ LOW RISK — This customer currently appears unlikely "
                    "to churn."
                )

            with st.expander("View Model Input"):
                st.dataframe(
                    customer,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(
                "Prediction could not be completed. "
                "Please verify that the saved model was trained using "
                "the exact feature set shown below."
            )
            st.code(", ".join(MODEL_FEATURES))
            st.exception(e)

# ============================================================
# CUSTOMER RISK
# ============================================================

elif page == "👥 Customer Risk":

    st.markdown(
        '<div class="section-title">Customer Risk Center</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Score every customer using the saved model and prioritize "
        "customers for retention."
    )

    try:
        feature_df = prepare_model_input(df)

        probabilities = model.predict_proba(feature_df)[:, 1]

        risk_df = df.copy()
        risk_df["Churn Probability"] = probabilities
        risk_df["Risk Level"] = risk_df["Churn Probability"].apply(make_risk)

    except Exception as e:
        st.error(f"Could not score the dataset: {e}")
        st.stop()

    high_count = int((risk_df["Risk Level"] == "High").sum())
    medium_count = int((risk_df["Risk Level"] == "Medium").sum())
    low_count = int((risk_df["Risk Level"] == "Low").sum())

    a, b, c = st.columns(3)

    with a:
        st.metric("🚨 High Risk", f"{high_count:,}")

    with b:
        st.metric("⚠️ Medium Risk", f"{medium_count:,}")

    with c:
        st.metric("✅ Low Risk", f"{low_count:,}")

    st.markdown(
        '<div class="section-title">Risk Distribution</div>',
        unsafe_allow_html=True,
    )

    risk_counts = pd.DataFrame(
        {
            "Risk Level": ["High", "Medium", "Low"],
            "Customers": [high_count, medium_count, low_count],
        }
    )

    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Customers",
        text="Customers",
        title="Predicted Customer Risk Distribution",
    )
    fig.update_traces(textposition="outside")
    chart_layout(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Filters</div>',
        unsafe_allow_html=True,
    )

    f1, f2 = st.columns(2)

    with f1:
        selected_risk = st.multiselect(
            "Risk Level",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

    with f2:
        min_probability = st.slider(
            "Minimum Churn Probability",
            0.0,
            1.0,
            0.0,
            0.05,
        )

    filtered = risk_df[
        risk_df["Risk Level"].isin(selected_risk)
        & (risk_df["Churn Probability"] >= min_probability)
    ].copy()

    display_columns = [
        c for c in [
            "CustomerId",
            "Surname",
            "Geography",
            "Gender",
            "Age",
            "CreditScore",
            "Balance",
            "IsActiveMember",
            "Exited",
            "Churn Probability",
            "Risk Level",
        ]
        if c in filtered.columns
    ]

    table = filtered[display_columns].sort_values(
        "Churn Probability",
        ascending=False,
    )

    st.caption(f"Showing {min(len(table), 100):,} highest-risk matching customers.")

    st.dataframe(
        table.head(100),
        use_container_width=True,
        hide_index=True,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Risk Report",
        data=csv,
        file_name="customer_churn_risk_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.markdown(
        '<div class="section-title">Customer Analytics</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        gender_df = (
            df.groupby("Gender")[TARGET]
            .mean()
            .reset_index()
        )
        gender_df["Churn Rate"] = gender_df[TARGET] * 100

        fig = px.bar(
            gender_df,
            x="Gender",
            y="Churn Rate",
            text="Churn Rate",
            title="Churn Rate by Gender",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )
        chart_layout(fig, 350)
        fig.update_yaxes(title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        active_df = (
            df.groupby("IsActiveMember")[TARGET]
            .mean()
            .reset_index()
        )
        active_df["Churn Rate"] = active_df[TARGET] * 100
        active_df["Member Status"] = active_df["IsActiveMember"].map(
            {0: "Inactive", 1: "Active"}
        )

        fig = px.bar(
            active_df,
            x="Member Status",
            y="Churn Rate",
            text="Churn Rate",
            title="Churn Rate by Member Activity",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )
        chart_layout(fig, 350)
        fig.update_yaxes(title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    products_df = (
        df.groupby("NumOfProducts")[TARGET]
        .mean()
        .reset_index()
    )
    products_df["Churn Rate"] = products_df[TARGET] * 100

    fig = px.bar(
        products_df,
        x="NumOfProducts",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Number of Products",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    chart_layout(fig, 380)
    fig.update_yaxes(title="Churn Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="Age",
        y="Balance",
        color="Exited",
        title="Age vs Balance",
        opacity=0.65,
        labels={"Exited": "Churned"},
    )
    chart_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Additional Customer Insights</div>',
        unsafe_allow_html=True,
    )

    insight_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
        "Exited",
    ]

    corr = df[insight_cols].corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Feature Correlation Matrix",
        aspect="auto",
    )
    chart_layout(fig, 520)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Selected Model</div>
                <div class="kpi-value" style="font-size:21px;">
                    {model_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        value = (
            f"{float(roc_auc_meta):.3f}"
            if roc_auc_meta is not None
            else "N/A"
        )
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Saved ROC-AUC</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Training Rows</div>
                <div class="kpi-value">
                    {metadata.get("training_rows", "N/A")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-box">
            <b style="color:white;">Selected Model:</b> {model_name}
            <br><br>
            <b style="color:white;">Target:</b> Customer churn / Exited
            <br><br>
            <b style="color:white;">Classification threshold:</b>
            {metadata.get("classification_threshold", 0.50)}
            <br><br>
            <b style="color:white;">Features:</b>
            {", ".join(MODEL_FEATURES)}
            <br><br>
            The application uses the saved model to estimate churn
            probability and rank customers by risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    expected = get_model_expected_features(model)
    if expected:
        with st.expander("🔍 Saved model feature information"):
            st.write("Features exposed by the saved estimator:")
            st.code(", ".join(map(str, expected)))

    st.markdown(
        '<div class="section-title">Historical Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "These metrics evaluate the saved model on the loaded historical "
        "dataset. For a formal report, use a separate unseen test set."
    )

    try:
        feature_df = prepare_model_input(df)

        predictions = model.predict(feature_df)
        probabilities = model.predict_proba(feature_df)[:, 1]

        acc = accuracy_score(df[TARGET], predictions)
        precision = precision_score(
            df[TARGET], predictions, zero_division=0
        )
        recall = recall_score(
            df[TARGET], predictions, zero_division=0
        )
        f1 = f1_score(
            df[TARGET], predictions, zero_division=0
        )
        auc = roc_auc_score(df[TARGET], probabilities)

        p1, p2, p3, p4, p5 = st.columns(5)

        p1.metric("Accuracy", f"{acc:.1%}")
        p2.metric("Precision", f"{precision:.1%}")
        p3.metric("Recall", f"{recall:.1%}")
        p4.metric("F1 Score", f"{f1:.1%}")
        p5.metric("ROC-AUC", f"{auc:.3f}")

        cm = confusion_matrix(df[TARGET], predictions)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Stay", "Actual Churn"],
            columns=["Predicted Stay", "Predicted Churn"],
        )

        fig = px.imshow(
            cm_df,
            text_auto=True,
            title="Confusion Matrix",
            aspect="auto",
        )
        chart_layout(fig, 430)
        st.plotly_chart(fig, use_container_width=True)

        # Probability distribution
        probability_df = pd.DataFrame(
            {
                "Churn Probability": probabilities,
                "Actual Outcome": np.where(
                    df[TARGET].values == 1,
                    "Churned",
                    "Stayed",
                ),
            }
        )

        fig = px.histogram(
            probability_df,
            x="Churn Probability",
            color="Actual Outcome",
            nbins=25,
            title="Predicted Churn Probability Distribution",
            marginal="box",
        )
        chart_layout(fig, 430)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(
            "Could not calculate model evaluation metrics. "
            "Make sure the saved model was trained with the same "
            "feature set and compatible scikit-learn version."
        )
        st.exception(e)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ChurnAI • Customer Churn Intelligence Platform
        <br>
        Built with Python • Scikit-learn • Streamlit • Plotly
    </div>
    """,
    unsafe_allow_html=True,
)