import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🏭",
    layout="wide",
)


# =========================================================
# DATA LOADING
# The dashboard uses precomputed scenario outputs. It does not
# load the serialized ML model or unused recommendation/KPI files.
# =========================================================
@st.cache_data

def load_data():
    engineered = pd.read_csv("nassau_candy_engineered.csv")
    scenarios = pd.read_csv("nassau_candy_scenario_results.csv")
    return engineered, scenarios


try:
    engineered_df, scenario_df = load_data()
except FileNotFoundError as exc:
    st.error(
        f"Required data file not found: {exc.filename}. "
        "Keep nassau_candy_engineered.csv and "
        "nassau_candy_scenario_results.csv in the same GitHub folder as app.py."
    )
    st.stop()


ENGINEERED_REQUIRED = {
    "Product Name",
    "Region",
    "Ship Mode",
    "Current Factory",
    "Distance_km",
    "Lead Time",
}
SCENARIO_REQUIRED = {
    "Product Name",
    "Current Factory",
    "Scenario Factory",
    "Region",
    "Ship Mode",
    "Scenario Distance_km",
    "Distance Change_km",
    "Distance Change %",
    "Current Lead Time",
    "Predicted Lead Time",
    "Lead Time Reduction %",
    "Confidence Score",
}

missing_engineered = ENGINEERED_REQUIRED - set(engineered_df.columns)
missing_scenarios = SCENARIO_REQUIRED - set(scenario_df.columns)
if missing_engineered or missing_scenarios:
    messages = []
    if missing_engineered:
        messages.append(f"Engineered data is missing: {sorted(missing_engineered)}")
    if missing_scenarios:
        messages.append(f"Scenario data is missing: {sorted(missing_scenarios)}")
    st.error("Data schema validation failed. " + " ".join(messages))
    st.stop()


# Ensure numerical calculations work even if CSV type inference changes.
ENGINEERED_NUMERIC = ["Distance_km", "Lead Time"]
SCENARIO_NUMERIC = [
    "Scenario Distance_km",
    "Distance Change_km",
    "Distance Change %",
    "Current Lead Time",
    "Predicted Lead Time",
    "Lead Time Reduction %",
    "Confidence Score",
]

for column in ENGINEERED_NUMERIC:
    engineered_df[column] = pd.to_numeric(engineered_df[column], errors="coerce")
for column in SCENARIO_NUMERIC:
    scenario_df[column] = pd.to_numeric(scenario_df[column], errors="coerce")


# =========================================================
# GLOBAL SCORING + GUARDRAILS
# Scores are calculated against the FULL scenario dataset before
# applying dashboard filters. This keeps scores comparable across
# products, regions and ship modes.
# =========================================================
def global_percentile_score(values, reference_values, lower_q=0.01, upper_q=0.99):
    """Scale non-negative values to 0-100 using fixed global quantile bounds."""
    reference = pd.to_numeric(reference_values, errors="coerce")
    reference = reference.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0)
    current = pd.to_numeric(values, errors="coerce")
    current = current.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)

    if reference.empty:
        return pd.Series(0.0, index=values.index)

    low = float(reference.quantile(lower_q))
    high = float(reference.quantile(upper_q))

    if not np.isfinite(high) or high <= low:
        return pd.Series(0.0, index=values.index)

    return ((current - low) / (high - low)).clip(0, 1) * 100


scenarios_all = scenario_df.copy()
scenarios_all["Distance_Savings_km"] = (
    -scenarios_all["Distance Change_km"]
).clip(lower=0)
scenarios_all["Lead_Time_Benefit"] = (
    scenarios_all["Lead Time Reduction %"]
).clip(lower=0)

# Global score components: no historical gross-profit value is used as
# a substitute for scenario-specific profit impact.
scenarios_all["Lead_Time_Score"] = global_percentile_score(
    scenarios_all["Lead_Time_Benefit"],
    scenarios_all["Lead_Time_Benefit"],
)
scenarios_all["Logistics_Efficiency_Score"] = global_percentile_score(
    scenarios_all["Distance_Savings_km"],
    scenarios_all["Distance_Savings_km"],
)

# Risk flags are calculated on every alternative scenario and do not
# depend on the priority slider.
scenarios_all["Low_Confidence_Flag"] = scenarios_all["Confidence Score"] < 50
scenarios_all["Distance_Increase_Flag"] = scenarios_all["Distance Change_km"] >= 0
scenarios_all["Large_Distance_Increase_Flag"] = (
    scenarios_all["Distance Change %"] > 25
)
scenarios_all["No_Lead_Time_Improvement_Flag"] = (
    scenarios_all["Lead Time Reduction %"] <= 0
)
scenarios_all["High_Risk_Flag"] = (
    scenarios_all["Low_Confidence_Flag"]
    | scenarios_all["Distance_Increase_Flag"]
    | scenarios_all["No_Lead_Time_Improvement_Flag"]
)

# The eligibility test is stated directly. High_Risk_Flag remains a
# reporting field; it is not redundantly checked again here.
scenarios_all["Recommendation_Eligible"] = (
    (scenarios_all["Lead Time Reduction %"] > 0)
    & (scenarios_all["Distance Change_km"] < 0)
    & (scenarios_all["Confidence Score"] >= 50)
)


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("Optimization Controls")

products = sorted(engineered_df["Product Name"].dropna().unique().tolist())
regions = sorted(engineered_df["Region"].dropna().unique().tolist())
ship_modes = sorted(engineered_df["Ship Mode"].dropna().unique().tolist())

if not products:
    st.error("No products were found in the engineered dataset.")
    st.stop()

selected_product = st.sidebar.selectbox("Product", products)
selected_region = st.sidebar.selectbox("Region", ["All"] + regions)
selected_ship_mode = st.sidebar.selectbox("Ship Mode", ["All"] + ship_modes)

speed_vs_logistics = st.sidebar.slider(
    "Speed ↔ Logistics Efficiency Priority",
    min_value=0,
    max_value=100,
    value=50,
    step=10,
    help=(
        "0 = prioritize distance savings; 100 = prioritize "
        "model-predicted lead-time improvement. This is not a profit slider."
    ),
)

# Slider weights sum to 1. The score is calculated from globally scaled
# lead-time benefit and distance savings, not historical gross profit.
speed_weight = 0.15 + (0.70 * speed_vs_logistics / 100)
logistics_weight = 1.0 - speed_weight

scenarios_all["Recommendation Score"] = (
    speed_weight * scenarios_all["Lead_Time_Score"]
    + logistics_weight * scenarios_all["Logistics_Efficiency_Score"]
)


# =========================================================
# FILTERED DATA + DECISION-LEVEL RECOMMENDATIONS
# =========================================================
filtered_scenarios = scenarios_all[
    scenarios_all["Product Name"] == selected_product
].copy()

if selected_region != "All":
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios["Region"] == selected_region
    ].copy()

if selected_ship_mode != "All":
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios["Ship Mode"] == selected_ship_mode
    ].copy()

eligible_rows = filtered_scenarios[
    filtered_scenarios["Recommendation_Eligible"]
].copy()

if not eligible_rows.empty:
    filtered_recommendations = (
        eligible_rows
        .groupby(
            [
                "Product Name",
                "Current Factory",
                "Scenario Factory",
                "Region",
                "Ship Mode",
            ],
            as_index=False,
        )
        .agg(
            Avg_Lead_Time_Reduction_Pct=("Lead Time Reduction %", "mean"),
            Avg_Distance_Change_Pct=("Distance Change %", "mean"),
            Avg_Distance_Change_km=("Distance Change_km", "mean"),
            Avg_Confidence_Score=("Confidence Score", "mean"),
            Recommendation_Score=("Recommendation Score", "mean"),
            Scenario_Count=("Recommendation Score", "count"),
        )
        .sort_values("Recommendation_Score", ascending=False)
        .reset_index(drop=True)
    )
else:
    filtered_recommendations = pd.DataFrame(
        columns=[
            "Product Name",
            "Current Factory",
            "Scenario Factory",
            "Region",
            "Ship Mode",
            "Avg_Lead_Time_Reduction_Pct",
            "Avg_Distance_Change_Pct",
            "Avg_Distance_Change_km",
            "Avg_Confidence_Score",
            "Recommendation_Score",
            "Scenario_Count",
        ]
    )


# =========================================================
# TITLE + METHODOLOGY CONTEXT
# =========================================================
st.title("🏭 Nassau Candy Factory Optimization")
st.caption(
    "Factory reallocation, scenario simulation and recommendation system"
)
st.info(
    "Lead-time improvements are model-predicted scenario estimates, not "
    "verified delivery-time gains. The supplied historical Lead Time values "
    "have structural date anomalies. The predictive model also had weak test "
    "performance (R² ≈ 0.057). Validate any operational decision with real "
    "shipping records before implementation."
)

lead_distance_corr = scenarios_all[
    ["Distance Change %", "Lead Time Reduction %"]
].corr().iloc[0, 1]
st.caption(
    f"Method note: distance-change percentage and predicted lead-time "
    f"reduction have a dataset-level correlation of {lead_distance_corr:.3f}. "
    "The slider therefore ranks two separate objectives; it does not assume "
    "that shorter distance causes faster delivery."
)


# =========================================================
# KPI CARDS — DYNAMIC AND FILTER-CONTEXTUAL
# =========================================================
# KPI 1-3 are calculated for the selected product + region + ship mode.
# Coverage is calculated across all products under the selected region
# and ship-mode filters (not just the selected product).
if not filtered_recommendations.empty:
    kpi_lead_gain = filtered_recommendations[
        "Avg_Lead_Time_Reduction_Pct"
    ].mean()
    kpi_distance_variability = filtered_recommendations[
        "Avg_Distance_Change_km"
    ].std(ddof=0)
    if not np.isfinite(kpi_distance_variability):
        kpi_distance_variability = 0.0
    kpi_confidence = filtered_recommendations[
        "Avg_Confidence_Score"
    ].mean()
else:
    kpi_lead_gain = np.nan
    kpi_distance_variability = np.nan
    kpi_confidence = np.nan

coverage_scenarios = scenarios_all.copy()
if selected_region != "All":
    coverage_scenarios = coverage_scenarios[
        coverage_scenarios["Region"] == selected_region
    ]
if selected_ship_mode != "All":
    coverage_scenarios = coverage_scenarios[
        coverage_scenarios["Ship Mode"] == selected_ship_mode
    ]

coverage_total_products = coverage_scenarios["Product Name"].nunique()
coverage_products = coverage_scenarios.loc[
    coverage_scenarios["Recommendation_Eligible"], "Product Name"
].nunique()
coverage_pct = (
    (coverage_products / coverage_total_products) * 100
    if coverage_total_products
    else 0.0
)

all_scenario_mean_change = (
    filtered_scenarios["Lead Time Reduction %"].mean()
    if not filtered_scenarios.empty
    else np.nan
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Avg Lead-Time Gain",
    f"{kpi_lead_gain:.2f}%" if np.isfinite(kpi_lead_gain) else "—",
    help="Mean model-predicted lead-time reduction across eligible decision-level recommendations for the selected filters.",
)
kpi2.metric(
    "Distance-Savings Variability",
    f"{kpi_distance_variability:.1f} km" if np.isfinite(kpi_distance_variability) else "—",
    help="Standard deviation of average distance change across eligible recommendations. This is a logistics proxy, not profit-impact stability.",
)
kpi3.metric(
    "Avg Recommendation Confidence",
    f"{kpi_confidence:.1f}/100" if np.isfinite(kpi_confidence) else "—",
    help="Mean historical-support confidence across eligible decision-level recommendations for the selected filters.",
)
kpi4.metric(
    "Recommendation Coverage",
    f"{coverage_products}/{coverage_total_products} ({coverage_pct:.1f}%)",
    help="Share of products with at least one eligible recommendation under the selected region and ship-mode filters. Product selector does not change this card.",
)

st.caption(
    "KPI context — the average predicted lead-time change across ALL "
    f"alternative scenarios for the selected product/filters is "
    f"{all_scenario_mean_change:.2f}% (includes rejected scenarios). "
    "Positive values indicate predicted improvement; negative values indicate "
    "predicted deterioration. The distance variability card is not a monetary "
    "profit measure."
    if np.isfinite(all_scenario_mean_change)
    else "KPI context — no scenario rows match the selected product and filters."
)


# =========================================================
# MODULE 1 — FACTORY OPTIMIZATION SIMULATOR
# =========================================================
st.header("1. Factory Optimization Simulator")
st.write(
    f"Factory comparison for **{selected_product}** under the selected "
    "region and ship-mode filters."
)

baseline_orders = engineered_df[
    engineered_df["Product Name"] == selected_product
].copy()
if selected_region != "All":
    baseline_orders = baseline_orders[
        baseline_orders["Region"] == selected_region
    ].copy()
if selected_ship_mode != "All":
    baseline_orders = baseline_orders[
        baseline_orders["Ship Mode"] == selected_ship_mode
    ].copy()

if not baseline_orders.empty and not filtered_scenarios.empty:
    baseline_factory = baseline_orders["Current Factory"].mode().iloc[0]
    baseline_distance = baseline_orders["Distance_km"].mean()
    baseline_supplied_lead_time = baseline_orders["Lead Time"].mean()

    alternative_summary = (
        filtered_scenarios
        .groupby("Scenario Factory", as_index=False)
        .agg(
            Records=("Scenario Factory", "size"),
            Avg_Distance_km=("Scenario Distance_km", "mean"),
            Avg_Predicted_Lead_Time=("Predicted Lead Time", "mean"),
            Avg_Lead_Time_Reduction_Pct=("Lead Time Reduction %", "mean"),
            Avg_Confidence_Score=("Confidence Score", "mean"),
        )
        .rename(columns={"Scenario Factory": "Factory"})
    )
    alternative_summary["Assignment"] = "Alternative factory (scenario)"
    alternative_summary["Supplied_Current_Lead_Time"] = np.nan

    baseline_row = pd.DataFrame(
        [
            {
                "Factory": baseline_factory,
                "Assignment": "Current factory (historical baseline)",
                "Records": len(baseline_orders),
                "Avg_Distance_km": baseline_distance,
                "Avg_Predicted_Lead_Time": np.nan,
                "Avg_Lead_Time_Reduction_Pct": np.nan,
                "Avg_Confidence_Score": np.nan,
                "Supplied_Current_Lead_Time": baseline_supplied_lead_time,
            }
        ]
    )

    factory_comparison = pd.concat(
        [baseline_row, alternative_summary], ignore_index=True
    )
    factory_comparison = factory_comparison[
        [
            "Factory",
            "Assignment",
            "Records",
            "Avg_Distance_km",
            "Supplied_Current_Lead_Time",
            "Avg_Predicted_Lead_Time",
            "Avg_Lead_Time_Reduction_Pct",
            "Avg_Confidence_Score",
        ]
    ]

    st.dataframe(
        factory_comparison.style.format(
            {
                "Avg_Distance_km": "{:.1f}",
                "Supplied_Current_Lead_Time": "{:.1f}",
                "Avg_Predicted_Lead_Time": "{:.1f}",
                "Avg_Lead_Time_Reduction_Pct": "{:.2f}%",
                "Avg_Confidence_Score": "{:.1f}",
            },
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Baseline interpretation: the current-factory row reports observed "
        "historical distance and supplied Lead Time. Alternative-factory rows "
        "report scenario distance and model-predicted Lead Time. The supplied "
        "Lead Time is anomalous, so do not compare it as if it were a validated "
        "baseline forecast."
    )

    distance_chart_df = factory_comparison[
        ["Factory", "Assignment", "Avg_Distance_km"]
    ].copy()
    fig_distance = px.bar(
        distance_chart_df,
        x="Factory",
        y="Avg_Distance_km",
        color="Assignment",
        title="Average Distance: Current Assignment vs Alternative Factories",
        labels={"Avg_Distance_km": "Average distance (km)"},
    )
    st.plotly_chart(fig_distance, width="stretch")

    predicted_chart_df = alternative_summary.sort_values(
        "Avg_Predicted_Lead_Time", ascending=True
    )
    fig_predicted = px.bar(
        predicted_chart_df,
        x="Factory",
        y="Avg_Predicted_Lead_Time",
        title="Model-Predicted Lead Time Across Alternative Factories",
        labels={"Avg_Predicted_Lead_Time": "Predicted lead time (days)"},
    )
    st.plotly_chart(fig_predicted, width="stretch")
else:
    st.warning("No historical baseline or alternative scenarios match these filters.")


# =========================================================
# MODULE 2 — WHAT-IF SCENARIO ANALYSIS
# =========================================================
st.header("2. What-If Scenario Analysis")

if not filtered_recommendations.empty:
    best_scenario = filtered_recommendations.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Factory", best_scenario["Current Factory"])
    col2.metric("Recommended Factory", best_scenario["Scenario Factory"])
    col3.metric(
        "Predicted Lead-Time Gain",
        f"{best_scenario['Avg_Lead_Time_Reduction_Pct']:.2f}%",
    )
    col4.metric(
        "Average Distance Saved",
        f"{-best_scenario['Avg_Distance_Change_km']:.1f} km",
    )

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Model-predicted lead-time index",
                "Distance index",
            ],
            "Current assignment": [100.0, 100.0],
            "Recommended assignment": [
                max(0.0, 100.0 - best_scenario["Avg_Lead_Time_Reduction_Pct"]),
                max(0.0, 100.0 + best_scenario["Avg_Distance_Change_Pct"]),
            ],
        }
    )
    fig_comparison = px.bar(
        comparison_df,
        x="Metric",
        y=["Current assignment", "Recommended assignment"],
        barmode="group",
        title="Current vs Recommended Assignment (Current = 100 Index)",
        labels={"value": "Index", "variable": "Assignment"},
    )
    st.plotly_chart(fig_comparison, width="stretch")
    st.caption(
        "The chart uses index values (current assignment = 100), not absolute "
        "delivery days. Lead-time gains are model predictions; distance "
        "savings are calculated from factory/customer coordinates."
    )
else:
    st.warning(
        "No recommendation passes the current eligibility guardrails for "
        "this product and filter."
    )
    st.info(
        "Alternative factories were evaluated, but none has all three: "
        "positive model-predicted lead-time improvement, shorter distance, "
        "and confidence of at least 50/100. See the Risk & Impact Panel."
    )


# =========================================================
# MODULE 3 — RECOMMENDATION DASHBOARD
# =========================================================
st.header("3. Recommendation Dashboard")
st.caption(
    f"Ranking weights at the current slider setting: "
    f"{speed_weight:.0%} predicted speed and {logistics_weight:.0%} "
    "distance efficiency. Scores use fixed full-dataset scaling."
)

if not filtered_recommendations.empty:
    display_recommendations = filtered_recommendations.head(10).copy()
    display_recommendations = display_recommendations[
        [
            "Current Factory",
            "Scenario Factory",
            "Region",
            "Ship Mode",
            "Avg_Lead_Time_Reduction_Pct",
            "Avg_Distance_Change_km",
            "Avg_Confidence_Score",
            "Recommendation_Score",
            "Scenario_Count",
        ]
    ]
    st.dataframe(
        display_recommendations.style.format(
            {
                "Avg_Lead_Time_Reduction_Pct": "{:.2f}%",
                "Avg_Distance_Change_km": "{:.1f} km",
                "Avg_Confidence_Score": "{:.1f}",
                "Recommendation_Score": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
else:
    st.warning(
        "No eligible recommendations are available for the selected product and filters."
    )
    st.caption(
        "This means the evaluated alternatives did not pass the project's "
        "lead-time, distance and confidence guardrails. The app does not force "
        "a recommendation when none qualifies."
    )


# =========================================================
# MODULE 4 — RISK & LOGISTICS IMPACT PANEL
# =========================================================
st.header("4. Risk & Logistics Impact Panel")

if not filtered_scenarios.empty:
    low_confidence = filtered_scenarios["Low_Confidence_Flag"]
    distance_increase = filtered_scenarios["Distance_Increase_Flag"]
    no_lead_gain = filtered_scenarios["No_Lead_Time_Improvement_Flag"]
    high_risk = filtered_scenarios[filtered_scenarios["High_Risk_Flag"]].copy()
    eligible_count = int(filtered_scenarios["Recommendation_Eligible"].sum())

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("High-Risk Scenarios", f"{len(high_risk):,}")
    r2.metric("Low Confidence", f"{int(low_confidence.sum()):,}")
    r3.metric("Distance Increases", f"{int(distance_increase.sum()):,}")
    r4.metric("No Predicted Lead-Time Gain", f"{int(no_lead_gain.sum()):,}")
    r5.metric("Eligible Scenarios", f"{eligible_count:,}")

    st.caption(
        "Risk counts are based on all alternative-factory scenario rows after "
        "the selected product/region/ship-mode filters. Alert categories can "
        "overlap: one scenario may trigger more than one risk flag. The slider "
        "changes ranking, not risk counts or eligibility."
    )

    st.warning(
        "Monetary profit impact and profit-impact stability are not calculated. "
        "The dataset has no reassignment-specific shipping costs or factory "
        "production-cost changes. Distance savings are shown only as a logistics "
        "efficiency proxy; they must not be read as currency or profit."
    )

    if not high_risk.empty:
        high_risk["Risk Reason"] = high_risk.apply(
            lambda row: "; ".join(
                reason
                for flag, reason in [
                    (row["Low_Confidence_Flag"], "Confidence below 50/100"),
                    (row["Distance_Increase_Flag"], "Distance does not decrease"),
                    (row["Large_Distance_Increase_Flag"], "Distance increase exceeds 25%"),
                    (row["No_Lead_Time_Improvement_Flag"], "No predicted lead-time gain"),
                ]
                if bool(flag)
            ),
            axis=1,
        )

        risk_display = (
            high_risk.sort_values("Recommendation Score", ascending=False)
            .head(10)[
                [
                    "Current Factory",
                    "Scenario Factory",
                    "Region",
                    "Ship Mode",
                    "Lead Time Reduction %",
                    "Distance Change_km",
                    "Confidence Score",
                    "Recommendation Score",
                    "Risk Reason",
                ]
            ]
            .copy()
        )
        st.subheader("Highest-Ranked High-Risk Alternatives")
        st.dataframe(
            risk_display.style.format(
                {
                    "Lead Time Reduction %": "{:.2f}%",
                    "Distance Change_km": "{:.1f} km",
                    "Confidence Score": "{:.1f}",
                    "Recommendation Score": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("No high-risk scenarios were identified for these filters.")
else:
    st.info("No scenario data is available for the selected filters.")


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Nassau Candy Factory Reallocation & Shipping Optimization Recommendation System"
)
st.caption(
    "Analytical prototype — validate scenario estimates with operational "
    "shipping, cost and factory-capacity data before implementation."
)
