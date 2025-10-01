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

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Credit Card Portfolio Analysis")

# --- Data Loading (assuming the file is in the same directory as the script) ---
@st.cache_data  # Cache the data loading for better performance
def load_data():
    try:
        df = pd.read_excel("credit_card_portfolio_with_banks.xlsx")
        # Derived KPIs (guard against zero/division)
        df["profit_margin"] = df["net_profit"] / df["revenue"].replace(0, np.nan)
        df["ROI"] = df["net_profit"] / df["acquisition_cost"].replace(0, np.nan)
        df["payback_ratio"] = df["acquisition_cost"] / df["net_profit"].replace(0, np.nan)
        # Balance buckets for segmentation (duplicates='drop' handles tied edges)
        df["balance_bucket"] = pd.qcut(df["balance"], q=4, labels=["Low", "Mid-Low", "Mid-High", "High"], duplicates='drop')
        # Create 'default_flag' and 'total_cost' for cost analysis
        df["default_flag"] = (df["default_rate"] > 0).astype(int)
        df["total_cost"] = df["acquisition_cost"] + (df["balance"] * df["default_flag"])
        return df
    except FileNotFoundError:
        st.error("Error: 'credit_card_portfolio_with_banks.xlsx' not found. Please upload the file.")
        return pd.DataFrame()

df = load_data()

# If df is empty, stop further rendering
if df.empty:
    st.stop()

# Pre-compute some aggregates centrally so tabs can use them reliably
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
bank_kpis["profit_margin"] = bank_kpis["total_net_profit"] / bank_kpis["total_revenue"].replace(0, np.nan)
bank_kpis["ROI"] = bank_kpis["total_net_profit"] / bank_kpis["total_acq_cost"].replace(0, np.nan)

# ---------------------------------------------------------------------
# Create Tabs
# ---------------------------------------------------------------------
tabs = st.tabs([
    "Bank-Level Analysis", "Profit Concentration", "Segment Profitability",
    "Risk–Return Benchmarking", "Customer-Level Drivers", "Cohort Profitability",
    "Bank Concentration Risk", "Sensitivity Analysis", "Unit Economics Workflows",
    "Stress Testing Workflows", "Sensitivity Analysis (Advanced)", "Risk Measurement"
])

# ---------------- Tab 1 ----------------
with tabs[0]:
    st.header("Bank-Level Profitability Analysis")
    st.write(bank_kpis)

    st.subheader("Customer Distribution by Bank")
    fig, ax = plt.subplots(figsize=(8, 4))
    # Use a color palette sized to number of banks
    colors = sns.color_palette("Dark2", n_colors=len(bank_kpis))
    # Plot horizontal bar of customers per bank using precomputed values
    bank_kpis_sorted = bank_kpis.sort_values("customers", ascending=True)
    ax.barh(bank_kpis_sorted["bank_name"], bank_kpis_sorted["customers"], color=colors)
    # hide top/right spines properly
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("Number of Customers")
    st.pyplot(fig)

# ---------------- Tab 2 ----------------
with tabs[1]:
    st.header("Profit Concentration (Pareto Analysis)")
    df_sorted = df.sort_values("net_profit", ascending=False).reset_index(drop=True)
    df_sorted["cum_profit"] = df_sorted["net_profit"].cumsum()
    total_profit = df_sorted["net_profit"].sum() if not df_sorted["net_profit"].empty else 1
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
        customers=("customer_id", "count")
    ).reset_index()
    segment_pivot = segment_profit.pivot(index="bank_name", columns="balance_bucket", values="avg_profit")
    st.write("Average Net Profit per Customer by Bank and Balance Bucket:")
    st.dataframe(segment_pivot.fillna(0))

    segment_roi_pivot = segment_profit.pivot(index="bank_name", columns="balance_bucket", values="avg_roi")
    st.write("Average ROI per Customer by Bank and Balance Bucket:")
    st.dataframe(segment_roi_pivot.fillna(0))

# ---------------- Tab 4 ----------------
with tabs[3]:
    st.header("Risk–Return Benchmarking")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=bank_kpis,
        x="avg_default_rate",
        y="avg_net_profit",
        size="customers",
        hue="bank_name",
        sizes=(100, 1000),
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
        X = df[["balance", "acquisition_cost"]].copy()
        y = df["net_profit"].copy()
        X = sm.add_constant(X)
        model = sm.OLS(y, X, missing='drop').fit()
        st.write("Regression Results (Net Profit vs Balance and Acquisition Cost):")
        st.text(model.summary().as_text())

# ---------------- Tab 6 ----------------
with tabs[5]:
    st.header("Cohort Profitability (Bank-level proxy)")
    cohort_profit = df.groupby("bank_name").agg({
        "net_profit": "mean",
        "ROI": "mean",
        "default_rate": "mean"
    }).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cohort_profit["bank_name"], cohort_profit["net_profit"], marker="o", label="Net Profit")
    ax.plot(cohort_profit["bank_name"], cohort_profit["ROI"], marker="s", label="ROI")
    ax.set_title("Cohort Profitability by Bank")
    ax.set_xlabel("Bank")
    ax.legend()
    st.pyplot(fig)

# ---------------- Tab 7 ----------------
with tabs[6]:
    st.header("Bank Concentration Risk")
    bank_profit = df.groupby("bank_name")["net_profit"].sum()
    share = bank_profit / bank_profit.sum()
    HHI = (share**2).sum() * 10000
    st.write(f"Herfindahl Index (HHI): {HHI:.2f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    bank_profit.plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_title("Profit Share by Bank")
    ax.set_ylabel("")
    st.pyplot(fig)

# ---------------- Tab 8 ----------------
with tabs[7]:
    st.header("Sensitivity Analysis (Simplified)")
    scenarios = {
        "Base": df["net_profit"].sum(),
        "Default +10%": (df["net_profit"] - 0.1 * df["balance"]).sum(),
        "Spend -15%": (df["revenue"] * 0.85 - df["acquisition_cost"]).sum(),
        "Cost +20%": (df["revenue"] - df["acquisition_cost"] * 1.2).sum()
    }
    scenario_df = pd.DataFrame.from_dict(scenarios, orient="index", columns=["Portfolio Profit"])

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
    discount_rate = 0.10
    lifetime_years = 3
    profit_per_customer = df.groupby("customer_id")["net_profit"].sum().mean()
    CLV = sum([profit_per_customer / ((1 + discount_rate) ** t) for t in range(1, lifetime_years + 1)])
    st.write(f"Average Customer Lifetime Value (CLV): {CLV:.2f}")

    st.subheader("Workflow 4: Payback Period Analysis")
    df["monthly_profit"] = df["net_profit"] / 12
    df["payback_months"] = df["acquisition_cost"] / (df["monthly_profit"].replace(0, np.nan))
    avg_payback = df["payback_months"].mean()
    st.write(f"Average Payback Period: {avg_payback:.1f} months")
    fig, ax = plt.subplots()
    ax.hist(df["payback_months"].dropna(), bins=30, color="teal", edgecolor="black")
    ax.set_title("Distribution of Payback Periods (Months)")
    ax.set_xlabel("Months")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("Workflow 5: Break-even Analysis")
    df["profit_margin_rate"] = (df["revenue"] - df["total_cost"]) / df["revenue"].replace(0, np.nan)
    df["required_revenue"] = df["total_cost"] / df["profit_margin_rate"].replace(0, np.nan)
    avg_required_revenue = df["required_revenue"].mean()
    st.write(f"Average Break-even Revenue Required: {avg_required_revenue:.2f}")
    fig, ax = plt.subplots()
    ax.scatter(df["revenue"], df["required_revenue"], alpha=0.4)
    ax.plot([0, max(df["revenue"].max(), df["required_revenue"].max())],
            [0, max(df["revenue"].max(), df["required_revenue"].max())],
            linestyle="--", color="red", label="Revenue = Required Revenue")
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
        profit = (df["revenue"] * (1 + shocks["spend_shock"])
                  - (df["acquisition_cost"] * (1 + shocks["cost_shock"]))
                  - (df["balance"] * shocks["default_shock"])).sum()
        ROI = profit / (df["total_cost"].sum())
        results.append([scenario, profit, ROI])
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
    ax.barh(sens_df.index, sens_df["Profit Change"], color="orange")
    ax.axvline(0, color="black")
    ax.set_title("Sensitivity Analysis (Tornado Style)")
    st.pyplot(fig)

# ---------------- Tab 11 ----------------
with tabs[10]:
    st.header("Sensitivity Analysis (Advanced)")
    st.subheader("Sensitivity to Interest Rate Changes")
    rate_changes = np.arange(-0.02, 0.05, 0.01)
    sensitivity = []
    for change in rate_changes:
        stressed_profit = (df["revenue"] * (1 + (change / df["interest_rate"].mean())) - df["acquisition_cost"] - df["loss"]).sum()
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

    def simulate_once():
        default_shock = np.random.uniform(0.05, 0.20)
        spend_shock = np.random.uniform(-0.20, 0.05)
        cost_shock = np.random.uniform(0.00, 0.20)
        profit = (df["revenue"] * (1 + spend_shock)
                  - (df["acquisition_cost"] * (1 + cost_shock))
                  - (df["balance"] * default_shock)).sum()
        return profit

    n = 1000
    profits = [simulate_once() for _ in range(n)]
    fig, ax = plt.subplots()
    ax.hist(profits, bins=40, color="steelblue", edgecolor="black")
    ax.set_title("Monte Carlo Simulation of Portfolio Profits")
    ax.set_xlabel("Profit")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    st.write(f"Expected Profit: {np.mean(profits):.2f}")
    st.write(f"5th percentile (Worst-case): {np.percentile(profits,5):.2f}")
    st.write(f"95th percentile (Best-case): {np.percentile(profits,95):.2f}")

    st.subheader("Value at Risk (VaR)")
    VaR_95 = np.percentile(df['net_profit'], 5)
    st.write(f"Portfolio Value at Risk (95% confidence): {round(VaR_95, 2)}")

    st.subheader("Stress Scenario Summary")
    base_profit = df['net_profit'].sum()
    stressed_revenue_interest = df['revenue'] * (1 + (0.02 / df['interest_rate'].mean()))
    profit_interest_shock = (stressed_revenue_interest - df['acquisition_cost'] - df['loss']).sum()
    stressed_loss_default = df['loss'] * 1.5
    profit_default_shock = (df['revenue'] - df['acquisition_cost'] - stressed_loss_default).sum()
    stressed_revenue_combined = df['revenue'] * (1 + (0.02 / df['interest_rate'].mean()))
    stressed_loss_combined = df['loss'] * 1.5
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
