import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: handle missing statsmodels gracefully
try:
    import statsmodels.api as sm
    STATS_MODELS_AVAILABLE = True
except Exception:
    STATS_MODELS_AVAILABLE = False

# Streamlit Page Setup
st.set_page_config(layout="wide", page_title="Credit Card Portfolio Analysis")

# ---------------- Data Loading ----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("credit_card_portfolio_with_banks.xlsx")
    except FileNotFoundError:
        return pd.DataFrame()

    # Derived KPIs with safe division
    df["profit_margin"] = df["net_profit"] / df["revenue"].replace(0, np.nan)
    df["ROI"] = df["net_profit"] / df["acquisition_cost"].replace(0, np.nan)
    df["payback_ratio"] = df["acquisition_cost"] / df["net_profit"].replace(0, np.nan)

    # Balance buckets
    try:
        df["balance_bucket"] = pd.qcut(df["balance"], q=4,
                                       labels=["Low", "Mid-Low", "Mid-High", "High"],
                                       duplicates="drop")
    except Exception:
        df["balance_bucket"] = "Unknown"

    # Flags and costs
    df["default_flag"] = (df["default_rate"] > 0).astype(int)
    df["total_cost"] = df["acquisition_cost"] + (df["balance"] * df["default_flag"])

    # Replace infinities/NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

df = load_data()

# If no data, allow user upload
if df.empty:
    uploaded = st.file_uploader("Upload Credit Card Portfolio Excel", type=["xlsx"])
    if uploaded is not None:
        df = pd.read_excel(uploaded)
    else:
        st.warning("Please upload 'credit_card_portfolio_with_banks.xlsx' to continue.")
        st.stop()

# ---------------- Aggregates ----------------
bank_kpis = df.groupby("bank_name").agg(
    customers=("customer_id", "count"),
    total_balance=("balance", "sum"),
    total_revenue=("revenue", "sum"),
    total_acq_cost=("acquisition_cost", "sum"),
    total_loss=("loss", "sum"),
    total_net_profit=("net_profit", "sum"),
    avg_net_profit=("net_profit", "mean"),
    avg_default_rate=("default_rate", "mean"),
    avg_interest_rate=("interest_rate", "mean")
).reset_index()

bank_kpis["profit_margin"] = (
    bank_kpis["total_net_profit"] / bank_kpis["total_revenue"].replace(0, np.nan)
)
bank_kpis["ROI"] = (
    bank_kpis["total_net_profit"] / bank_kpis["total_acq_cost"].replace(0, np.nan)
)

bank_kpis = bank_kpis.replace([np.inf, -np.inf], np.nan).fillna(0)

# ---------------- Tabs ----------------
tabs = st.tabs([
    "Bank-Level Analysis", "Profit Concentration", "Segment Profitability",
    "Risk–Return Benchmarking", "Customer-Level Drivers", "Cohort Profitability",
    "Bank Concentration Risk", "Sensitivity Analysis", "Unit Economics Workflows",
    "Stress Testing Workflows", "Sensitivity Analysis (Advanced)", "Risk Measurement"
])

# ---------------- Tab 1 ----------------
with tabs[0]:
    st.header("Bank-Level Profitability Analysis")
    st.dataframe(bank_kpis)

    st.subheader("Customer Distribution by Bank")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("Dark2", n_colors=len(bank_kpis))
    bank_kpis_sorted = bank_kpis.sort_values("customers", ascending=True)
    ax.barh(bank_kpis_sorted["bank_name"], bank_kpis_sorted["customers"], color=colors)
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("Bank")
    st.pyplot(fig)

# ---------------- Tab 2 ----------------
with tabs[1]:
    st.header("Profit Concentration (Pareto Analysis)")
    if "net_profit" in df:
        df_sorted = df.sort_values("net_profit", ascending=False).reset_index(drop=True)
        df_sorted["cum_profit"] = df_sorted["net_profit"].cumsum()
        total_profit = df_sorted["net_profit"].sum() or 1
        df_sorted["cum_profit_pct"] = df_sorted["cum_profit"] / total_profit
        df_sorted["cum_customers_pct"] = (df_sorted.index + 1) / len(df_sorted)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_sorted["cum_customers_pct"], df_sorted["cum_profit_pct"], marker=".")
        ax.axhline(0.8, color="r", linestyle="--", label="80% Profit Line")
        ax.axvline(0.2, color="g", linestyle="--", label="20% Customers Line")
        ax.set_title("Profit Concentration Curve (Pareto)")
        ax.set_xlabel("Cumulative % of Customers")
        ax.set_ylabel("Cumulative % of Total Net Profit")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Net profit column not available for Pareto analysis.")
