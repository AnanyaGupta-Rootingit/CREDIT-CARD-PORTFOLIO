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

# Streamlit page config (must be called before other UI calls)
st.set_page_config(layout="wide", page_title="Credit Card Portfolio Analysis")

REQUIRED_COLUMNS = [
    "customer_id", "bank_name", "balance", "revenue", "acquisition_cost",
    "net_profit", "default_rate", "interest_rate", "loss"
]

@st.cache_data
def read_excel_file(file_bytes):
    """Read uploaded excel file or local file bytes into a DataFrame."""
    try:
        return pd.read_excel(file_bytes)
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        return pd.DataFrame()


def safe_divide(a, b):
    """Return a / b with zeros handled (returns NaN for 0 denominator)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.array(a) / np.array(b)
    return pd.Series(res).replace([np.inf, -np.inf], np.nan)


# ----------------- Data loading UI -----------------
st.sidebar.header("Data input")
use_local = st.sidebar.checkbox("Try local default filename first (credit_card_portfolio_with_banks.xlsx)", value=True)
uploaded = st.sidebar.file_uploader("Upload your credit card portfolio Excel file", type=["xlsx", "xls"]) 

@st.cache_data
def load_data_from_source(use_local_flag, uploaded_file):
    df = pd.DataFrame()
    if use_local_flag:
        try:
            df = pd.read_excel("credit_card_portfolio_with_banks.xlsx")
        except FileNotFoundError:
            # ignore, fall back to uploader
            df = pd.DataFrame()
        except Exception:
            df = pd.DataFrame()

    if df.empty and uploaded_file is not None:
        df = read_excel_file(uploaded_file)

    return df

# load
df = load_data_from_source(use_local, uploaded)

if df.empty:
    st.warning("No data loaded yet. Please upload the Excel file or enable local file option if file exists on disk.")
    st.stop()

# Basic validation
missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    st.error(f"The dataset is missing required columns: {missing_cols}")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Ensure numeric types for required numeric columns
numeric_cols = ["balance", "revenue", "acquisition_cost", "net_profit", "default_rate", "interest_rate", "loss"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Derived KPIs (guard against zero/division)
df["profit_margin"] = safe_divide(df["net_profit"], df["revenue"].replace(0, np.nan))
df["ROI"] = safe_divide(df["net_profit"], df["acquisition_cost"].replace(0, np.nan))
df["payback_ratio"] = safe_divide(df["acquisition_cost"], df["net_profit"].replace(0, np.nan))

# Create default flag and total cost
# If default_rate > 0 we'll consider a potential default indicator, but keep original default_rate for calculations
df["default_flag"] = (df["default_rate"] > 0).astype(int)
df["total_cost"] = df["acquisition_cost"].fillna(0) + (df["balance"].fillna(0) * df["default_flag"])

# Balance buckets: handle case where there are fewer distinct values than q
try:
    n_quantiles = min(4, df["balance"].nunique())
    if n_quantiles >= 2:
        labels = ["Low", "Mid-Low", "Mid-High", "High"][:n_quantiles]
        df["balance_bucket"] = pd.qcut(df["balance"].rank(method='first'), q=n_quantiles, labels=labels, duplicates='drop')
    else:
        df["balance_bucket"] = "Single"
except Exception as e:
    df["balance_bucket"] = "Unknown"

# Pre-compute bank-level KPIs
bank_kpis = df.groupby("bank_name").agg(
    customers=("customer_id", "nunique"),
    total_balance=("balance", "sum"),
    total_revenue=("revenue", "sum"),
    total_acq_cost=("acquisition_cost", "sum"),
    total_loss=("loss", "sum"),
    total_net_profit=("net_profit", "sum"),
    avg_net_profit=("net_profit", "mean"),
    avg_default_rate=("default_rate", "mean"),
    avg_interest_rate=("interest_rate", "mean")
).reset_index()

bank_kpis["profit_margin"] = safe_divide(bank_kpis["total_net_profit"], bank_kpis["total_revenue"].replace(0, np.nan))
bank_kpis["ROI"] = safe_divide(bank_kpis["total_net_profit"], bank_kpis["total_acq_cost"].replace(0, np.nan))

# ----------- UI Tabs -----------
TAB_NAMES = [
    "Bank-Level Analysis", "Profit Concentration", "Segment Profitability",
    "Risk–Return Benchmarking", "Customer-Level Drivers", "Cohort Profitability",
    "Bank Concentration Risk", "Sensitivity Analysis", "Unit Economics Workflows",
    "Stress Testing Workflows", "Sensitivity Analysis (Advanced)", "Risk Measurement"
]

tabs = st.tabs(TAB_NAMES)

# ---------------- Tab 1 ----------------
with tabs[0]:
    st.header("Bank-Level Profitability Analysis")
    st.dataframe(bank_kpis)

    st.subheader("Customer Distribution by Bank")
    if not bank_kpis.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        bank_kpis_sorted = bank_kpis.sort_values("customers", ascending=True)
        palette = sns.color_palette("Dark2", n_colors=max(2, len(bank_kpis_sorted)))
        ax.barh(bank_kpis_sorted["bank_name"], bank_kpis_sorted["customers"], color=palette[:len(bank_kpis_sorted)])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Number of Customers")
        st.pyplot(fig)
    else:
        st.info("No bank KPI rows to plot.")

# ---------------- Tab 2 ----------------
with tabs[1]:
    st.header("Profit Concentration (Pareto Analysis)")
    df_sorted = df.sort_values("net_profit", ascending=False).reset_index(drop=True)
    df_sorted["cum_profit"] = df_sorted["net_profit"].cumsum()
    total_profit = df_sorted["net_profit"].sum()
    if total_profit == 0:
        st.warning("Total net profit is zero — Pareto curve will be meaningless.")
        total_profit = 1
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

# ---------------- Tab 3 ----------------
with tabs[2]:
    st.header("Segment Profitability (Balance Buckets)")
    segment_profit = df.groupby(["bank_name", "balance_bucket"]).agg(
        avg_profit=("net_profit", "mean"),
        avg_roi=("ROI", "mean"),
        customers=("customer_id", "nunique")
    ).reset_index()

    if not segment_profit.empty:
        segment_pivot = segment_profit.pivot(index="bank_name", columns="balance_bucket", values="avg_profit")
        st.write("Average Net Profit per Customer by Bank and Balance Bucket:")
        st.dataframe(segment_pivot.fillna(0))

        segment_roi_pivot = segment_profit.pivot(index="bank_name", columns="balance_bucket", values="avg_roi")
        st.write("Average ROI per Customer by Bank and Balance Bucket:")
        st.dataframe(segment_roi_pivot.fillna(0))
    else:
        st.info("No segment profit data to show.")

# ---------------- Tab 4 ----------------
with tabs[3]:
    st.header("Risk–Return Benchmarking")
    if bank_kpis.empty:
        st.info("Not enough bank-level KPI data to plot risk-return.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=bank_kpis,
            x="avg_default_rate",
            y="avg_net_profit",
            size="customers",
            hue="bank_name",
            sizes=(50, 800),
            alpha=0.7,
            ax=ax,
            legend=False
        )
        ax.set_title("Risk vs Return Benchmarking (Bank-Level)")
        ax.set_xlabel("Average Default Rate")
        ax.set_ylabel("Average Net Profit per Customer")
        st.pyplot(fig)

# ---------------- Tab 5 ----------------
with tabs[4]:
    st.header("Customer-Level Profitability Drivers")
    if not STATS_MODELS_AVAILABLE:
        st.warning("statsmodels not installed — regression table will not be shown. Install statsmodels to enable regression.")
    else:
        # Basic regression: net_profit ~ balance + acquisition_cost
        X = df[["balance", "acquisition_cost"]].copy()
        y = df["net_profit"].copy()
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X, missing='drop').fit()
            st.write("Regression Results (Net Profit vs Balance and Acquisition Cost):")
            st.text(model.summary().as_text())
        except Exception as e:
            st.error(f"Regression failed: {e}")

# ---------------- Tab 6 ----------------
with tabs[5]:
    st.header("Cohort Profitability (Bank-level proxy)")
    cohort_profit = df.groupby("bank_name").agg({
        "net_profit": "mean",
        "ROI": "mean",
        "default_rate": "mean"
    }).reset_index()

    if not cohort_profit.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cohort_profit["bank_name"], cohort_profit["net_profit"], marker="o", label="Net Profit")
        ax.plot(cohort_profit["bank_name"], cohort_profit["ROI"], marker="s", label="ROI")
        ax.set_title("Cohort Profitability by Bank")
        ax.set_xlabel("Bank")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No cohort data to plot.")

# ---------------- Tab 7 ----------------
with tabs[6]:
    st.header("Bank Concentration Risk")
    bank_profit = df.groupby("bank_name")["net_profit"].sum()
    if bank_profit.sum() == 0:
        st.warning("Total profit is zero — HHI will be meaningless.")
    share = safe_divide(bank_profit, bank_profit.sum().replace(0, np.nan))
    HHI = (share.fillna(0) ** 2).sum() * 10000
    st.write(f"Herfindahl Index (HHI): {HHI:.2f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    bank_profit.plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_title("Profit Share by Bank")
    ax.set_ylabel("")
    st.pyplot(fig)

# ---------------- Tab 8 ----------------
with tabs[7]:
    st.header("Sensitivity Analysis (Simplified)")
    base = df["net_profit"].sum()
    scenarios_simple = {
        "Base": base,
        "Default +10%": (df["net_profit"] - 0.1 * df["balance"]).sum(),
        "Spend -15%": (df["revenue"] * 0.85 - df["acquisition_cost"]).sum(),
        "Cost +20%": (df["revenue"] - df["acquisition_cost"] * 1.2).sum()
    }
    scenario_df = pd.DataFrame.from_dict(scenarios_simple, orient="index", columns=["Portfolio Profit"]) 
    fig, ax = plt.subplots(figsize=(7, 5))
    scenario_df.plot(kind="bar", ax=ax, legend=False)
    ax.set_title("Sensitivity Analysis: Portfolio Profit Impact")
    ax.set_ylabel("Profit")
    st.pyplot(fig)

# ---------------- Tab 9 ----------------
with tabs[8]:
    st.header("Unit Economics Workflows")
    st.subheader("Workflow 1: Revenue per Customer")
    rev_per_customer = df.groupby("customer_id")["revenue"].sum().mean()
    rev_per_card = df["revenue"].mean()
    st.write(f"Average Revenue per Customer: {rev_per_customer:.2f}")
    st.write(f"Average Revenue per Card: {rev_per_card:.2f}")
    bank_rev = df.groupby("bank_name")["revenue"].mean().sort_values(ascending=False)
    if not bank_rev.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        bank_rev.plot(kind="bar", ax=ax, title="Average Revenue per Card (Bank-wise)")
        ax.set_ylabel("Revenue")
        st.pyplot(fig)

    st.subheader("Workflow 2: Cost per Customer")
    avg_cost_per_customer = df.groupby("customer_id")["total_cost"].sum().mean()
    CAC = df["acquisition_cost"].mean()
    st.write(f"Average Total Cost per Customer: {avg_cost_per_customer:.2f}")
    st.write(f"Customer Acquisition Cost (CAC): {CAC:.2f}")
    cost_breakdown = {
        "Acquisition Cost": df["acquisition_cost"].mean(),
        "Risk Cost": (df["balance"] * df["default_flag"]).mean()
    }
    fig, ax = plt.subplots()
    ax.bar(cost_breakdown.keys(), cost_breakdown.values())
    ax.set_title("Average Cost Components per Customer")
    ax.set_ylabel("Cost")
    st.pyplot(fig)

    st.subheader("Workflow 3: Customer Lifetime Value (CLV)")
    discount_rate = st.number_input("Discount rate (decimal)", value=0.10, min_value=0.0, max_value=1.0, step=0.01)
    lifetime_years = st.number_input("Lifetime (years)", value=3, min_value=1, max_value=10, step=1)
    profit_per_customer = df.groupby("customer_id")["net_profit"].sum().mean()
    CLV = sum([profit_per_customer / ((1 + discount_rate) ** t) for t in range(1, int(lifetime_years) + 1)])
    st.write(f"Average Customer Lifetime Value (CLV): {CLV:.2f}")

    st.subheader("Workflow 4: Payback Period Analysis")
    df["monthly_profit"] = df["net_profit"] / 12
    df["payback_months"] = safe_divide(df["acquisition_cost"], df["monthly_profit"].replace(0, np.nan))
    avg_payback = df["payback_months"].replace([np.inf, -np.inf], np.nan).mean()
    st.write(f"Average Payback Period: {avg_payback:.1f} months")
    fig, ax = plt.subplots()
    ax.hist(df["payback_months"].dropna(), bins=30)
    ax.set_title("Distribution of Payback Periods (Months)")
    ax.set_xlabel("Months")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("Workflow 5: Break-even Analysis")
    df["profit_margin_rate"] = safe_divide((df["revenue"] - df["total_cost"]), df["revenue"].replace(0, np.nan))
    df["required_revenue"] = safe_divide(df["total_cost"], df["profit_margin_rate"].replace(0, np.nan))
    avg_required_revenue = df["required_revenue"].mean()
    st.write(f"Average Break-even Revenue Required: {avg_required_revenue:.2f}")
    fig, ax = plt.subplots()
    max_val = max(df["revenue"].max(skipna=True) or 0, df["required_revenue"].max(skipna=True) or 0)
    ax.scatter(df["revenue"], df["required_revenue"].replace([np.inf, -np.inf], np.nan), alpha=0.4)
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="red", label="Revenue = Required Revenue")
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Required Revenue for Break-even")
    ax.set_title("Break-even Revenue Analysis")
    ax.legend()
    st.pyplot(fig)

# ---------------- Tab 10 ----------------
with tabs[9]:
    st.header("Stress Testing Workflows")
    st.subheader("Workflow 1: Baseline vs Shock Comparison")
    baseline_profit = df["net_profit"].sum()
    scenarios = {
        "Baseline": baseline_profit,
        "Default +10%": (df["net_profit"] - 0.10 * df["balance"]).sum(),
        "Spend -15%": (df["revenue"] * 0.85 - df["total_cost"]).sum(),
        "Acquisition Cost +20%": (df["revenue"] - (df["acquisition_cost"] * 1.2)).sum()
    }
    scenario_df = pd.DataFrame.from_dict(scenarios, orient="index", columns=["Portfolio Profit"])
    st.write(scenario_df)

    st.subheader("Workflow 2: Multi-Scenario Profitability Table")
    multi_scenarios = {
        "Mild": {"default_shock": 0.05, "spend_shock": -0.05, "cost_shock": 0.05},
        "Moderate": {"default_shock": 0.10, "spend_shock": -0.10, "cost_shock": 0.10},
        "Severe": {"default_shock": 0.20, "spend_shock": -0.20, "cost_shock": 0.20}
    }
    results = []
    for scenario, shocks in multi_scenarios.items():
        profit = (df["revenue"] * (1 + shocks["spend_shock"]) - (df["acquisition_cost"] * (1 + shocks["cost_shock"]) ) - (df["balance"] * shocks["default_shock"]) ).sum()
        ROI = safe_divide(profit, df["total_cost"].sum().replace(0, np.nan))
        results.append([scenario, profit, float(ROI) if not pd.isna(ROI) else np.nan])
    multi_table = pd.DataFrame(results, columns=["Scenario", "Profit", "ROI"])
    st.write(multi_table)

    st.subheader("Workflow 3: Stress Testing Defaults")
    default_rates = [0.05, 0.10, 0.15, 0.20]
    stress_results = []
    for dr in default_rates:
        stressed_profit = (df["net_profit"] - dr * df["balance"]).sum()
        stress_results.append([dr, stressed_profit])
    default_stress_df = pd.DataFrame(stress_results, columns=["Extra Default Rate", "Portfolio Profit"])
    fig, ax = plt.subplots()
    ax.plot(default_stress_df["Extra Default Rate"] * 100, default_stress_df["Portfolio Profit"], marker="o")
    ax.set_title("Portfolio Profit Under Rising Defaults")
    ax.set_xlabel("Increase in Default Rate (%)")
    ax.set_ylabel("Portfolio Profit")
    st.pyplot(fig)

    st.subheader("Workflow 4: Sensitivity Analysis (Tornado Style)")
    base_profit = df["net_profit"].sum()
    sensitivities = {
        "Defaults +10%": (df["net_profit"] - 0.1 * df["balance"]).sum() - base_profit,
        "Spend -15%": (df["revenue"] * 0.85 - df["total_cost"]).sum() - base_profit,
        "Acquisition Cost +20%": (df["revenue"] - (df["acquisition_cost"] * 1.2)).sum() - base_profit,
    }
    sens_df = pd.DataFrame.from_dict(sensitivities, orient="index", columns=["Profit Change"]).sort_values("Profit Change")
    fig, ax = plt.subplots()
    ax.barh(sens_df.index, sens_df["Profit Change"]) 
    ax.axvline(0, color="black")
    ax.set_title("Sensitivity Analysis (Tornado Style)")
    st.pyplot(fig)

# ---------------- Tab 11 ----------------
with tabs[10]:
    st.header("Sensitivity Analysis (Advanced)")
    st.subheader("Sensitivity to Interest Rate Changes")
    rate_changes = np.arange(-0.02, 0.05, 0.01)
    sensitivity = []
    avg_interest = df['interest_rate'].mean()
    if pd.isna(avg_interest) or avg_interest == 0:
        st.warning("Interest rate mean is zero or NaN — results may be unstable.")
        avg_interest = 1.0

    for change in rate_changes:
        stressed_profit = (df["revenue"] * (1 + (change / avg_interest)) - df["acquisition_cost"] - df["loss"]).sum()
        sensitivity.append({"rate_change": round(change, 3), "portfolio_profit": stressed_profit})
    sensitivity_df = pd.DataFrame(sensitivity)
    st.write(sensitivity_df)
    fig, ax = plt.subplots()
    ax.plot(sensitivity_df["rate_change"] * 100, sensitivity_df["portfolio_profit"], marker="o")
    ax.set_title("Portfolio Profit Sensitivity to Interest Rate Changes")
    ax.set_xlabel("Change in Interest Rate (%)")
    ax.set_ylabel("Portfolio Profit")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Sensitivity to Default Rate Changes")
    default_changes = np.arange(0.8, 1.6, 0.1)
    sensitivity = []
    for factor in default_changes:
        stressed_profit = (df['revenue'] - df['acquisition_cost'] - (df['loss'] * factor)).sum()
        sensitivity.append({"default_factor": round(factor, 2), "portfolio_profit": stressed_profit})
    sensitivity_df = pd.DataFrame(sensitivity)
    st.write(sensitivity_df)
    fig, ax = plt.subplots()
    ax.plot(sensitivity_df["default_factor"] * 100, sensitivity_df["portfolio_profit"], marker="o")
    ax.set_title("Portfolio Profit Sensitivity to Default Rate Changes")
    ax.set_xlabel("Default Rate Factor (%)")
    ax.set_ylabel("Portfolio Profit")
    ax.grid(True)
    st.pyplot(fig)

# ---------------- Tab 12 ----------------
with tabs[11]:
    st.header("Risk Measurement")
    st.subheader("Monte Carlo Simulation of Portfolio Profits")

    def simulate_once(df_local):
        default_shock = np.random.uniform(0.05, 0.20)
        spend_shock = np.random.uniform(-0.20, 0.05)
        cost_shock = np.random.uniform(0.00, 0.20)
        profit = (df_local["revenue"] * (1 + spend_shock)
                  - (df_local["acquisition_cost"] * (1 + cost_shock))
                  - (df_local["balance"] * default_shock)).sum()
        return profit

    n = st.number_input("Monte Carlo runs", min_value=100, max_value=50000, value=1000, step=100)
    profits = [simulate_once(df) for _ in range(int(n))]
    fig, ax = plt.subplots()
    ax.hist(profits, bins=40)
    ax.set_title("Monte Carlo Simulation of Portfolio Profits")
    ax.set_xlabel("Profit")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    st.write(f"Expected Profit: {np.mean(profits):.2f}")
    st.write(f"5th percentile (Worst-case): {np.percentile(profits,5):.2f}")
    st.write(f"95th percentile (Best-case): {np.percentile(profits,95):.2f}")

    st.subheader("Value at Risk (VaR)")
    VaR_95 = np.percentile(df['net_profit'].dropna(), 5) if len(df['net_profit'].dropna())>0 else np.nan
    st.write(f"Portfolio Value at Risk (95% confidence): {round(VaR_95, 2)}")

    st.subheader("Stress Scenario Summary")
    base_profit = df['net_profit'].sum()
    avg_interest = df['interest_rate'].mean()
    if pd.isna(avg_interest) or avg_interest == 0:
        avg_interest = 1.0
    stressed_revenue_interest = df['revenue'] * (1 + (0.02 / avg_interest))
    profit_interest_shock = (stressed_revenue_interest - df['acquisition_cost'] - df['loss']).sum()
    stressed_loss_default = df['loss'] * 1.5
    profit_default_shock = (df['revenue'] - df['acquisition_cost'] - stressed_loss_default).sum()
    stressed_revenue_combined = stressed_revenue_interest
    stressed_loss_combined = stressed_loss_default
    profit_combined = (stressed_revenue_combined - df['acquisition_cost'] - stressed_loss_combined).sum()
    summary = {
        "Base Profit": base_profit,
        "Interest Shock Profit": profit_interest_shock,
        "Default Shock Profit": profit_default_shock,
        "Combined Stress Profit": profit_combined
    }
    summary_df = pd.DataFrame(list(summary.items()), columns=["Scenario", "Portfolio Profit"])
    st.write(summary_df)
    fig, ax = plt.subplots()
    summary_df.set_index("Scenario").plot(kind="bar", legend=False, ax=ax)
    ax.set_title("Portfolio Profit Under Different Stress Scenarios")
    ax.set_ylabel("Portfolio Profit")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# End of script
