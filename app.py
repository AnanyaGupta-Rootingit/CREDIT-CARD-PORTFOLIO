import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import base64, io

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Credit Card Portfolio Analytics"

# Layout
app.layout = dbc.Container([
    html.H1("📊 Credit Card Portfolio Dashboard", className="mt-3 mb-4"),

    # Upload
    dcc.Upload(
        id='upload-data',
        children=html.Div(["Drag and Drop or ", html.A("Select a CSV File")]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
               'borderWidth': '1px', 'borderStyle': 'dashed',
               'borderRadius': '5px', 'textAlign': 'center'},
        multiple=False
    ),

    html.Hr(),

    # Tabs
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Portfolio Overview", value="overview"),
        dcc.Tab(label="Bank KPIs", value="kpis"),
        dcc.Tab(label="Risk vs Return", value="risk"),
        dcc.Tab(label="Pareto Curve", value="pareto"),
        dcc.Tab(label="CLV Analysis", value="clv"),
        dcc.Tab(label="Unit Economics", value="unit"),
        dcc.Tab(label="Segment Benchmarking", value="segment"),
        dcc.Tab(label="Monte Carlo Simulation", value="montecarlo"),
        dcc.Tab(label="Sensitivity (Tornado)", value="tornado"),
        dcc.Tab(label="HHI Analysis", value="hhi"),
        dcc.Tab(label="Payback Distribution", value="payback"),
        dcc.Tab(label="Raw Data", value="raw"),
    ]),

    html.Div(id="tab-content", style={"marginTop": "20px"})
])

# --- Helper to parse file ---
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

# --- Main callback ---
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    State("upload-data", "contents"),
)
def render_tab(tab, contents):
    if contents is None:
        return html.Div("📂 Upload a CSV file to start.")

    df = parse_contents(contents)

    # ---- 1. Overview ----
    if tab == "overview":
        return html.Div([
            html.H3("Portfolio Overview"),
            html.P(f"Total Customers: {df['customers'].sum():,}"),
            html.P(f"Average Net Profit: {df['net_profit'].mean():,.2f}"),
            dcc.Graph(figure=px.histogram(df, x="net_profit", nbins=20,
                                          title="Net Profit Distribution"))
        ])

    # ---- 2. Bank KPIs ----
    elif tab == "kpis":
        bank_kpis = df.groupby("bank_name").agg({
            "customers": "sum",
            "net_profit": "mean",
            "default_rate": "mean"
        }).reset_index()
        return dcc.Graph(
            figure=px.bar(bank_kpis, x="customers", y="bank_name", color="bank_name",
                          orientation="h", title="Customer Distribution by Bank",
                          hover_data=["net_profit", "default_rate"])
        )

    # ---- 3. Risk vs Return ----
    elif tab == "risk":
        return dcc.Graph(
            figure=px.scatter(df, x="default_rate", y="net_profit", size="customers",
                              color="bank_name", hover_data=["customers"],
                              title="Risk vs Return Benchmarking")
        )

    # ---- 4. Pareto Curve ----
    elif tab == "pareto":
        df_sorted = df.sort_values("customers", ascending=False)
        df_sorted["cum_customers"] = df_sorted["customers"].cumsum() / df_sorted["customers"].sum()
        df_sorted["rank"] = range(1, len(df_sorted) + 1)
        return dcc.Graph(
            figure=px.line(df_sorted, x="rank", y="cum_customers",
                           title="Pareto Curve - Customer Concentration")
        )

    # ---- 5. CLV ----
    elif tab == "clv":
        df["CLV"] = df["net_profit"] * (1 - df["default_rate"])
        return dcc.Graph(
            figure=px.bar(df, x="CLV", y="bank_name", color="bank_name",
                          orientation="h", title="Customer Lifetime Value by Bank")
        )

    # ---- 6. Unit Economics ----
    elif tab == "unit":
        df["unit_econ"] = df["net_profit"] - (df["default_rate"] * df["net_profit"])
        return dcc.Graph(
            figure=px.bar(df, x="unit_econ", y="bank_name", color="bank_name",
                          orientation="h", title="Unit Economics by Bank")
        )

    # ---- 7. Segment Benchmarking ----
    elif tab == "segment":
        if "segment" in df.columns:
            return dcc.Graph(
                figure=px.bar(df, x="segment", y="net_profit", color="bank_name",
                              barmode="group", title="Segment Benchmarking",
                              hover_data=["customers", "default_rate"])
            )
        else:
            return html.Div("⚠️ No 'segment' column found in dataset.")

    # ---- 8. Monte Carlo ----
    elif tab == "montecarlo":
        def simulate_once():
            sampled = df.sample(frac=1, replace=True)
            return sampled["net_profit"].sum()
        profits = [simulate_once() for _ in range(1000)]
        return dcc.Graph(
            figure=px.histogram(profits, nbins=40,
                                title="Monte Carlo Simulation of Portfolio Profits")
        )

    # ---- 9. Tornado ----
    elif tab == "tornado":
        scenarios = {
            "Default Rate +10%": df["net_profit"].mean() * (1 - (df["default_rate"].mean() * 1.1)),
            "Default Rate -10%": df["net_profit"].mean() * (1 - (df["default_rate"].mean() * 0.9)),
            "Profit +10%": df["net_profit"].mean() * 1.1,
            "Profit -10%": df["net_profit"].mean() * 0.9
        }
        sens_df = pd.DataFrame(list(scenarios.items()), columns=["Scenario", "Profit"])
        return dcc.Graph(
            figure=px.bar(sens_df, x="Profit", y="Scenario", orientation="h",
                          title="Tornado Sensitivity Analysis")
        )

    # ---- 10. HHI ----
    elif tab == "hhi":
        bank_kpis = df.groupby("bank_name")["customers"].sum().reset_index()
        bank_kpis["market_share"] = bank_kpis["customers"] / bank_kpis["customers"].sum()
        bank_kpis["share_sq"] = bank_kpis["market_share"] ** 2
        hhi = bank_kpis["share_sq"].sum() * 10000
        return html.Div([
            html.H3(f"HHI Index: {hhi:.0f}"),
            dcc.Graph(
                figure=px.bar(bank_kpis, x="market_share", y="bank_name",
                              orientation="h", title="Market Share by Bank")
            )
        ])

    # ---- 11. Payback ----
    elif tab == "payback":
        if "payback_period" in df.columns:
            return dcc.Graph(
                figure=px.histogram(df, x="payback_period", nbins=20,
                                    title="Payback Period Distribution")
            )
        else:
            return html.Div("⚠️ No 'payback_period' column found in dataset.")

    # ---- 12. Raw Data ----
    elif tab == "raw":
        return dbc.Table.from_dataframe(df.head(20), striped=True, bordered=True, hover=True)

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
