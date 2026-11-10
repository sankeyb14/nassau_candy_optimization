
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🏭",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    engineered_df = pd.read_csv(
        "nassau_candy_engineered.csv"
    )

    scenario_df = pd.read_csv(
        "nassau_candy_scenario_results.csv"
    )

    recommendation_df = pd.read_csv(
        "nassau_candy_recommendation_summary.csv"
    )

    top_recommendations = pd.read_csv(
        "nassau_candy_top_recommendations.csv"
    )

    kpis = pd.read_csv(
        "nassau_candy_kpis.csv"
    )

    return (
        engineered_df,
        scenario_df,
        recommendation_df,
        top_recommendations,
        kpis
    )

    return (
        engineered_df,
        scenario_df,
        recommendation_df,
        top_recommendations,
        kpis
    )


@st.cache_resource
def load_model():
    return joblib.load(
        "/content/nassau_candy_final_model.joblib"
    )


(
    engineered_df,
    scenario_df,
    recommendation_df,
    top_recommendations,
    kpis
) = load_data()

model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Optimization Controls")

products = sorted(
    engineered_df["Product Name"].dropna().unique()
)

regions = sorted(
    engineered_df["Region"].dropna().unique()
)

ship_modes = sorted(
    engineered_df["Ship Mode"].dropna().unique()
)

selected_product = st.sidebar.selectbox(
    "Product",
    products
)

selected_region = st.sidebar.selectbox(
    "Region",
    ["All"] + regions
)

selected_ship_mode = st.sidebar.selectbox(
    "Ship Mode",
    ["All"] + ship_modes
)

speed_vs_profit = st.sidebar.slider(
    "Speed ↔ Profit Priority",
    min_value=0,
    max_value=100,
    value=50,
    step=10,
    help="0 = prioritize profit value, 100 = prioritize speed."
)


# =========================================================
# TITLE
# =========================================================

st.title("🏭 Nassau Candy Factory Optimization")
st.caption(
    "Factory reallocation, scenario simulation and recommendation system"
)

st.info(
    "Important: Lead-time improvements shown here are model-predicted "
    "scenario estimates. The supplied historical Lead Time variable "
    "contains structural date anomalies and should not be interpreted "
    "as verified real-world shipping duration."
)


# =========================================================
# KPI CARDS
# =========================================================

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Lead-Time Reduction",
    f"{kpis.loc[0, 'Value']:.2f}%"
)

kpi2.metric(
    "Impact Stability",
    f"{kpis.loc[1, 'Value']:.2f}"
)

kpi3.metric(
    "Scenario Confidence",
    f"{kpis.loc[2, 'Value']:.2f}"
)

kpi4.metric(
    "Recommendation Coverage",
    f"{kpis.loc[3, 'Value']:.2f}%"
)


# =========================================================
# FILTERED DATA
# =========================================================

filtered_scenarios = scenario_df[
    scenario_df["Product Name"] == selected_product
].copy()

filtered_recommendations = recommendation_df[
    recommendation_df["Product Name"] == selected_product
].copy()

if selected_region != "All":
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios["Region"] == selected_region
    ]

    filtered_recommendations = filtered_recommendations[
        filtered_recommendations["Region"] == selected_region
    ]

if selected_ship_mode != "All":
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios["Ship Mode"] == selected_ship_mode
    ]

    filtered_recommendations = filtered_recommendations[
        filtered_recommendations["Ship Mode"] == selected_ship_mode
    ]


# =========================================================
# MODULE 1
# FACTORY OPTIMIZATION SIMULATOR
# =========================================================

st.header("1. Factory Optimization Simulator")

st.write(
    f"Predicted performance for **{selected_product}** "
    f"across alternative factories."
)

factory_summary = (
    filtered_scenarios
    .groupby("Scenario Factory", as_index=False)
    .agg(
        Avg_Predicted_Lead_Time=(
            "Predicted Lead Time",
            "mean"
        ),
        Avg_Distance=(
            "Scenario Distance_km",
            "mean"
        ),
        Avg_Lead_Time_Reduction=(
            "Lead Time Reduction %",
            "mean"
        ),
        Avg_Confidence=(
            "Confidence Score",
            "mean"
        )
    )
    .sort_values(
        "Avg_Predicted_Lead_Time"
    )
)

if len(factory_summary) > 0:

    st.dataframe(
        factory_summary.style.format({
            "Avg_Predicted_Lead_Time": "{:.1f}",
            "Avg_Distance": "{:.1f} km",
            "Avg_Lead_Time_Reduction": "{:.2f}%",
            "Avg_Confidence": "{:.1f}"
        }),
        use_container_width=True
    )

    fig_factory = px.bar(
        factory_summary,
        x="Scenario Factory",
        y="Avg_Predicted_Lead_Time",
        title="Predicted Lead Time by Factory"
    )

    st.plotly_chart(
        fig_factory,
        use_container_width=True
    )


# =========================================================
# MODULE 2
# WHAT-IF SCENARIO ANALYSIS
# =========================================================

st.header("2. What-If Scenario Analysis")

if len(filtered_recommendations) > 0:

    best_scenario = (
        filtered_recommendations
        .sort_values(
            "Recommendation_Score",
            ascending=False
        )
        .iloc[0]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Factory",
        best_scenario["Current Factory"]
    )

    col2.metric(
        "Recommended Factory",
        best_scenario["Scenario Factory"]
    )

    col3.metric(
        "Lead-Time Improvement",
        f"{best_scenario['Avg_Lead_Time_Reduction_Pct']:.2f}%"
    )

    col4.metric(
        "Distance Change",
        f"{best_scenario['Avg_Distance_Change_Pct']:.2f}%"
    )

    comparison_df = pd.DataFrame({
        "Metric": [
            "Lead Time",
            "Distance"
        ],
        "Current": [
            100,
            100
        ],
        "Recommended": [
            100 - best_scenario[
                "Avg_Lead_Time_Reduction_Pct"
            ],
            100 + best_scenario[
                "Avg_Distance_Change_Pct"
            ]
        ]
    })

    fig_comparison = px.bar(
        comparison_df,
        x="Metric",
        y=["Current", "Recommended"],
        barmode="group",
        title="Current vs Recommended Assignment"
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )

else:
    st.warning(
        "No eligible recommendation matches the selected filters."
    )


# =========================================================
# MODULE 3
# RECOMMENDATION DASHBOARD
# =========================================================

st.header("3. Recommendation Dashboard")

if len(filtered_recommendations) > 0:

    display_recommendations = (
        filtered_recommendations
        .sort_values(
            "Recommendation_Score",
            ascending=False
        )
        .head(10)
        .copy()
    )

    display_recommendations = display_recommendations[
        [
            "Current Factory",
            "Scenario Factory",
            "Region",
            "Ship Mode",
            "Avg_Lead_Time_Reduction_Pct",
            "Avg_Distance_Change_Pct",
            "Avg_Confidence_Score",
            "Recommendation_Score",
            "Scenario_Count"
        ]
    ]

    st.dataframe(
        display_recommendations.style.format({
            "Avg_Lead_Time_Reduction_Pct": "{:.2f}%",
            "Avg_Distance_Change_Pct": "{:.2f}%",
            "Avg_Confidence_Score": "{:.1f}",
            "Recommendation_Score": "{:.2f}"
        }),
        use_container_width=True
    )

else:
    st.warning(
        "No recommendations available for the selected filters."
    )


# =========================================================
# MODULE 4
# RISK & IMPACT PANEL
# =========================================================

st.header("4. Risk & Impact Panel")

if len(filtered_recommendations) > 0:

    low_confidence = (
        filtered_recommendations[
            filtered_recommendations[
                "Avg_Confidence_Score"
            ] < 50
        ]
    )

    distance_increase = (
        filtered_recommendations[
            filtered_recommendations[
                "Avg_Distance_Change_Pct"
            ] > 25
        ]
    )

    st.subheader("Risk Alerts")

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Low-Confidence Options",
        len(low_confidence)
    )

    r2.metric(
        "Large Distance Increase",
        len(distance_increase)
    )

    r3.metric(
        "Eligible Recommendations",
        len(filtered_recommendations)
    )

    st.warning(
        "Monetary profit impact is not directly estimated because "
        "the dataset does not contain reassignment-specific shipping "
        "cost data. Distance change is used as the operational "
        "logistics-impact proxy."
    )

    if len(low_confidence) > 0:

        st.subheader("Low-Confidence Recommendations")

        st.dataframe(
            low_confidence[
                [
                    "Current Factory",
                    "Scenario Factory",
                    "Region",
                    "Ship Mode",
                    "Avg_Confidence_Score",
                    "Recommendation_Score"
                ]
            ],
            use_container_width=True
        )

else:
    st.info(
        "No recommendation risk data available for the selected filters."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Nassau Candy Factory Reallocation & Shipping Optimization "
    "Recommendation System"
)

st.caption(
    "Analytical prototype — recommendations should be validated "
    "against operational shipping and cost data before implementation."
)
