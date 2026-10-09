import streamlit as st
import pandas as pd
import joblib
from web_agent import web_search
import json
from forecast_tools import *

from scenario_engine import (
    apply_business_scenario
)

from feature_engineering import (
    engineer_features,
    build_future_features
)

from forecast_engine import (
    create_future_calendar,
    add_scenario_features,
    forecast_quantity,
    forecast_sales,
    forecast_profit
)

from sales_features import (
    SALES_CATEGORICAL_COLS,
    SALES_NUMERICAL_COLS
)

from quantity_feature import (
    QUANTITY_CATEGORICAL_COLS,
    QUANTITY_NUMERICAL_COLS
)

from profit_features import (
    PROFIT_CATEGORICAL_COLS,
    PROFIT_NUMERICAL_COLS
)

from business_agent import (
    validate_business_data,
    generate_business_summary
)

from parser import (
    detect_intent
)

from chatbot import (
    ask_llm
)

from forecasting_agent import (
    run_forecast
)

from data_intake_agent import (
    standardize_dataset
)

from data_cleaner import (
    clean_dataset
)


if "messages" not in st.session_state:

    st.session_state.messages = []

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title(
    "AI Business Intelligence Assistant"
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

sales_model = joblib.load("sales_model.pkl")
quantity_model = joblib.load("quantity_model.pkl")
profit_model = joblib.load("profit_model.pkl")

# --------------------------------------------------
# FEATURE LISTS
# --------------------------------------------------

sales_features = (
    SALES_CATEGORICAL_COLS +
    SALES_NUMERICAL_COLS
)

quantity_features = (
    QUANTITY_CATEGORICAL_COLS +
    QUANTITY_NUMERICAL_COLS
)

profit_features = (
    PROFIT_CATEGORICAL_COLS +
    PROFIT_NUMERICAL_COLS
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.info(
    """
Upload your business sales spreadsheet.

Required Fields:

• Date
• Sales
• Quantity
• Profit
• State
• Category
• Sub Category
• Discount
• Order_date


The system will automatically clean
and prepare the data before forecasting.
"""
)

uploaded_file = st.file_uploader(
    "Upload Business Data",
    type=["csv"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # AI schema mapping
    df = standardize_dataset(df)

    st.subheader(
        "Detected Schema"
    )

    st.dataframe(
        pd.DataFrame(
            {"Detected Columns": df.columns}
        )
    )

    # cleaning
    df = clean_dataset(df)

    # validation
    missing = validate_business_data(df)

    if missing:
        st.error(
            f"Missing columns: {missing}"
        )
        st.stop()

    summary = generate_business_summary(df)

    st.success(summary)

    st.subheader(
        "Data Quality Report"
    )

    duplicates = df.duplicated().sum()

    missing_sales = (
        df["sales"].isna().sum()
        if "sales" in df.columns
        else 0
    )

    missing_profit = (
        df["profit"].isna().sum()
        if "profit" in df.columns
        else 0
    )

    missing_quantity = (
        df["quantity"].isna().sum()
        if "quantity" in df.columns
        else 0
    )

    st.write(
        f"""
    Rows Uploaded:
    {len(df):,}

    Duplicate Rows:
    {duplicates}

    Missing Sales:
    {missing_sales}

    Missing Profit:
    {missing_profit}

    Missing Quantity:
    {missing_quantity}

    Date Range:
    {df['order_date'].min()}
    to
    {df['order_date'].max()}
    """
    )

    summary = f"""
    Dataset loaded successfully.

    Rows:
    {len(df)}

    States:
    {df['state'].nunique()}

    Categories:
    {df['category'].nunique()}

    Date Range:
    {df['order_date'].min()}
    to
    {df['order_date'].max()}
    """

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    historical_df = engineer_features(df)

    if st.checkbox("Show Debug Data"):
        st.write(
            historical_df.groupby(
                ["state","category","sub_category"]
            ).size().reset_index(name="rows")
        )

    # --------------------------------------------------
    # BUSINESS SELECTION
    # --------------------------------------------------

    state = st.selectbox(
        "Select State",
        sorted(
            historical_df["state"].unique()
        )
    )

    category = st.selectbox(
        "Select Category",
        sorted(
            historical_df["category"].unique()
        )
    )

    sub_category = st.selectbox(
        "Select Sub Category",
        sorted(
            historical_df["sub_category"].unique()
        )
    )

    # --------------------------------------------------
    # SCENARIO INPUTS
    # --------------------------------------------------

    st.subheader("Business Scenario")

    price_growth_pct = st.slider(
        "Expected Price Growth (%)",
        -50,
        100,
        5
    )

    discount_pct = st.slider(
        "Expected Discount (%)",
        0,
        100,
        10
    )

    demand_growth_pct = st.slider(
        "Expected Demand Growth (%)",
        -50,
        100,
        15
    )

    promotion_flag = st.selectbox(
        "Promotion Planned?",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    economic_outlook = st.selectbox(
        "Economic Outlook",
        [
            "negative",
            "neutral",
            "positive"
        ]
    )

    scenario = {
        "price_growth_pct": price_growth_pct,
        "discount_pct": discount_pct,
        "demand_growth_pct": demand_growth_pct,
        "promotion_flag": promotion_flag,
        "economic_outlook": economic_outlook
    }

    st.subheader("Forecast Horizon")

    forecast_mode = st.radio(
        "Forecast Type",
        [
            "Year Range",
            "Custom Months"
        ]
    )

    if forecast_mode == "Year Range":

        start_year = st.number_input(
            "Start Year",
            min_value=2024,
            max_value=2050,
            value=2024
        )

        end_year = st.number_input(
            "End Year",
            min_value=2024,
            max_value=2050,
            value=2026
        )

        start_date = pd.Timestamp(
            f"{start_year}-01-01"
        )

        end_date = pd.Timestamp(
            f"{end_year}-12-31"
        )

    else:

        start_date = st.date_input(
            "Forecast Start Month",
            value=pd.Timestamp("2024-01-01")
        )

        end_date = st.date_input(
            "Forecast End Month",
            value=pd.Timestamp("2025-12-01")
        )

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)


    # validation for BOTH modes
    if end_date <= start_date:

        st.error(
            "End date must be after start date."
        )

        st.stop()
    # --------------------------------------------------
    # GENERATE FORECAST
    # --------------------------------------------------

    if st.button("Generate Forecast"):

        business_df = historical_df[
            (historical_df["state"] == state)
            &
            (historical_df["category"] == category)
            &
            (historical_df["sub_category"] == sub_category)
        ]

        if business_df.empty:

            st.warning(
                f"""
                No historical records were found for:

                • State: {state}
                • Category: {category}
                • Sub Category: {sub_category}

                Please verify that the selected sub-category
                belongs to the selected category and state.
                """
            )

            st.stop()

        historical_end = (
            business_df["order_date"].max()
            + pd.offsets.MonthEnd(1)
        )

        forecast_start = max(
            historical_end,
            start_date
        )

        st.info(
            f"Forecast will begin from "
            f"{forecast_start.strftime('%Y-%m')}"
        )

        forecast_months = (
            (end_date.year - forecast_start.year) * 12
            +
            (end_date.month - forecast_start.month)
        ) + 1

        if forecast_months <= 0:

            st.error(
                "Forecast end date must be after the forecast start date."
            )

            st.stop()

        future_df = create_future_calendar(
            start_date=forecast_start,
            forecast_months=forecast_months
        )

        future_df = build_future_features(
            future_df,
            business_df
        )

        future_df["state"] = state
        future_df["category"] = category
        future_df["sub_category"] = sub_category

        future_df = add_scenario_features(
            future_df,
            scenario
        )

        future_df = forecast_quantity(
            future_df,
            quantity_model,
            quantity_features
        )

        future_df = forecast_sales(
            future_df,
            sales_model,
            sales_features
        )

        future_df = forecast_profit(
            future_df,
            profit_model,
            profit_features
        )

        baseline_df = future_df.copy()

        scenario_df = apply_business_scenario(
            baseline_df.copy(),
            scenario,
        )

        # --------------------------------------------------
        # KPI CALCULATIONS
        # --------------------------------------------------

        baseline_sales = float(
            baseline_df["sales"].sum()
        )

        scenario_sales = float(
            scenario_df["sales"].sum()
        )

        baseline_profit = float(
            baseline_df["profit"].sum()
        )

        scenario_profit = float(
            scenario_df["profit"].sum()
        )

        baseline_quantity = float(
            baseline_df["quantity"].sum()
        )

        scenario_quantity = float(
            scenario_df["quantity"].sum()
        )

        sales_impact = (
            scenario_sales -
            baseline_sales
        )

        profit_impact = (
            scenario_profit -
            baseline_profit
        )

        quantity_impact = (
            scenario_quantity -
            baseline_quantity
        )

        # --------------------------------------------------
        # MONTHLY FORECAST
        # --------------------------------------------------

        monthly_forecast = pd.DataFrame({

            "Month": pd.to_datetime(
                baseline_df["order_date"]
            ).dt.strftime("%Y-%m"),

            "Baseline Sales":
                baseline_df["sales"],

            "Scenario Sales":
                scenario_df["sales"],

            "Baseline Quantity":
                baseline_df["quantity"],

            "Scenario Quantity":
                scenario_df["quantity"],

            "Baseline Profit":
                baseline_df["profit"],

            "Scenario Profit":
                scenario_df["profit"]

        })

        # --------------------------------------------------
        # YEARLY SUMMARY
        # --------------------------------------------------

        yearly_summary = monthly_forecast.copy()

        yearly_summary["Year"] = (
            pd.to_datetime(
                yearly_summary["Month"]
            ).dt.year
        )

        yearly_summary = (
            yearly_summary
            .groupby("Year")
            .sum(numeric_only=True)
            .reset_index()
        )

        # --------------------------------------------------
        # SAVE EVERYTHING
        # --------------------------------------------------

        st.session_state["forecast_generated"] = True

        st.session_state["baseline_df"] = baseline_df

        st.session_state["scenario_df"] = scenario_df

        st.session_state["monthly_forecast_df"] = monthly_forecast

        st.session_state["yearly_summary_df"] = yearly_summary

        st.session_state["forecast_results"] = {

            "baseline_sales": baseline_sales,
            "scenario_sales": scenario_sales,

            "baseline_profit": baseline_profit,
            "scenario_profit": scenario_profit,

            "baseline_quantity": baseline_quantity,
            "scenario_quantity": scenario_quantity,

            "sales_impact": sales_impact,
            "profit_impact": profit_impact,
            "quantity_impact": quantity_impact,

            "monthly_forecast":
                monthly_forecast.to_dict(
                    orient="records"
                ),

            "yearly_summary":
                yearly_summary.to_dict(
                    orient="records"
                )

        }

    def show_forecast_dashboard():

        baseline_df = st.session_state["baseline_df"]

        scenario_df = st.session_state["scenario_df"]

        monthly_forecast = st.session_state[
            "monthly_forecast_df"
        ]

        yearly_summary = st.session_state[
            "yearly_summary_df"
        ]

        results = st.session_state[
            "forecast_results"
        ]

        baseline_sales = results["baseline_sales"]
        scenario_sales = results["scenario_sales"]

        baseline_profit = results["baseline_profit"]
        scenario_profit = results["scenario_profit"]

        baseline_quantity = results["baseline_quantity"]
        scenario_quantity = results["scenario_quantity"]

        sales_impact = results["sales_impact"]
        profit_impact = results["profit_impact"]
        quantity_impact = results["quantity_impact"]

        # --------------------------------------------------
        # KPI CARDS
        # --------------------------------------------------

        st.subheader("Sales KPIs")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Baseline Sales",
            f"${baseline_sales:,.0f}"
        )

        col2.metric(
            "Scenario Sales",
            f"${scenario_sales:,.0f}"
        )

        col3.metric(
            "Impact",
            f"${sales_impact:,.0f}"
        )

        st.subheader("Quantity KPIs")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Baseline Quantity",
            f"{baseline_quantity:,.0f}"
        )

        col2.metric(
            "Scenario Quantity",
            f"{scenario_quantity:,.0f}"
        )

        col3.metric(
            "Impact",
            f"{quantity_impact:,.0f}"
        )

        st.subheader("Profit KPIs")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Baseline Profit",
            f"${baseline_profit:,.0f}"
        )

        col2.metric(
            "Scenario Profit",
            f"${scenario_profit:,.0f}"
        )

        col3.metric(
            "Impact",
            f"${profit_impact:,.0f}"
        )

        # --------------------------------------------------
        # MONTHLY FORECAST TABLE
        # --------------------------------------------------

        st.subheader(
            "Monthly Forecast Detail"
        )

        st.dataframe(
            monthly_forecast,
            use_container_width=True
        )

        csv = monthly_forecast.to_csv(
            index=False
        )

        st.download_button(
            label="Download Monthly Forecast",
            data=csv,
            file_name="monthly_forecast.csv",
            mime="text/csv"
        )

        # --------------------------------------------------
        # YEARLY SUMMARY
        # --------------------------------------------------

        st.subheader(
            "Yearly Summary"
        )

        st.dataframe(
            yearly_summary,
            use_container_width=True
        )

        # --------------------------------------------------
        # SALES CHART
        # --------------------------------------------------

        sales_chart = pd.DataFrame({

            "Baseline":
                baseline_df["sales"],

            "Scenario":
                scenario_df["sales"]

        })

        sales_chart.index = (
            baseline_df["order_date"]
        )

        st.subheader(
            "Sales Forecast"
        )

        st.line_chart(
            sales_chart
        )

        # --------------------------------------------------
        # QUANTITY CHART
        # --------------------------------------------------

        quantity_chart = pd.DataFrame({

            "Baseline":
                baseline_df["quantity"],

            "Scenario":
                scenario_df["quantity"]

        })

        quantity_chart.index = (
            baseline_df["order_date"]
        )

        st.subheader(
            "Quantity Forecast"
        )

        st.line_chart(
            quantity_chart
        )

        # --------------------------------------------------
        # PROFIT CHART
        # --------------------------------------------------

        profit_chart = pd.DataFrame({

            "Baseline":
                baseline_df["profit"],

            "Scenario":
                scenario_df["profit"]

        })

        profit_chart.index = (
            baseline_df["order_date"]
        )

        st.subheader(
            "Profit Forecast"
        )

        st.line_chart(
            profit_chart
        )

        # --------------------------------------------------
        # COMPARISON TABLE
        # --------------------------------------------------

        st.subheader(
            "Baseline vs Scenario Sales Forecast"
        )

        comparison_df = pd.DataFrame({

            "Baseline Sales":
                baseline_df["sales"],

            "Scenario Sales":
                scenario_df["sales"]

        })

        comparison_df.index = (
            baseline_df["order_date"]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

left, right = st.columns([2, 1])

with left:

    if st.session_state.get(
        "forecast_generated",
        False
    ):
        show_forecast_dashboard()

with right:

    st.header(
        "AI Business Consultant"
    )

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.write(
            msg["content"]
        )

import json

prompt = st.chat_input(
    "Ask a business question..."
)

def build_analysis_context():

    forecast = st.session_state.get(
        "forecast_results",
        {}
    )

    baseline_sales = forecast.get(
        "baseline_sales",
        0
    )

    scenario_sales = forecast.get(
        "scenario_sales",
        0
    )

    baseline_profit = forecast.get(
        "baseline_profit",
        0
    )

    scenario_profit = forecast.get(
        "scenario_profit",
        0
    )

    baseline_quantity = forecast.get(
        "baseline_quantity",
        0
    )

    scenario_quantity = forecast.get(
        "scenario_quantity",
        0
    )

    # -----------------------------
    # Impacts
    # -----------------------------

    sales_impact = (
        scenario_sales -
        baseline_sales
    )

    profit_impact = (
        scenario_profit -
        baseline_profit
    )

    quantity_impact = (
        scenario_quantity -
        baseline_quantity
    )

    # -----------------------------
    # Percentage Changes
    # -----------------------------

    sales_pct = (
        sales_impact /
        baseline_sales * 100
        if baseline_sales else 0
    )

    profit_pct = (
        profit_impact /
        baseline_profit * 100
        if baseline_profit else 0
    )

    quantity_pct = (
        quantity_impact /
        baseline_quantity * 100
        if baseline_quantity else 0
    )

    # -----------------------------
    # Profit Per Unit Analysis
    # -----------------------------

    profit_per_unit_baseline = (
        baseline_profit /
        baseline_quantity
        if baseline_quantity else 0
    )

    profit_per_unit_scenario = (
        scenario_profit /
        scenario_quantity
        if scenario_quantity else 0
    )

    profit_per_unit_change = (
        (
            profit_per_unit_scenario -
            profit_per_unit_baseline
        )
        / profit_per_unit_baseline * 100
        if profit_per_unit_baseline else 0
    )

    # -----------------------------
    # Revenue Per Unit Analysis
    # -----------------------------

    sales_per_unit_baseline = (
        baseline_sales /
        baseline_quantity
        if baseline_quantity else 0
    )

    sales_per_unit_scenario = (
        scenario_sales /
        scenario_quantity
        if scenario_quantity else 0
    )

    sales_per_unit_change = (
        (
            sales_per_unit_scenario -
            sales_per_unit_baseline
        )
        / sales_per_unit_baseline * 100
        if sales_per_unit_baseline else 0
    )

    profit_margin_baseline = (
        baseline_profit /
        baseline_sales * 100
        if baseline_sales else 0
    )

    profit_margin_scenario = (
        scenario_profit /
        scenario_sales * 100
        if scenario_sales else 0
    )

    profit_margin_change = (
        profit_margin_scenario -
        profit_margin_baseline
    )

    growth_efficiency = (
        sales_pct / quantity_pct
        if quantity_pct else 0
    )

    return f"""
Business Forecast Summary

Baseline Sales: ${baseline_sales:,.2f}
Scenario Sales: ${scenario_sales:,.2f}
Sales Impact: ${sales_impact:,.2f}
Sales Change: {sales_pct:.1f}%

Baseline Profit: ${baseline_profit:,.2f}
Scenario Profit: ${scenario_profit:,.2f}
Profit Impact: ${profit_impact:,.2f}
Profit Change: {profit_pct:.1f}%

Baseline Quantity: {baseline_quantity:,.2f}
Scenario Quantity: {scenario_quantity:,.2f}
Quantity Impact: {quantity_impact:,.2f}
Quantity Change: {quantity_pct:.1f}%

Profit Per Unit Baseline:
${profit_per_unit_baseline:.2f}

Profit Per Unit Scenario:
${profit_per_unit_scenario:.2f}

Profit Per Unit Change:
{profit_per_unit_change:.1f}%

Revenue Per Unit Baseline:
${sales_per_unit_baseline:.2f}

Revenue Per Unit Scenario:
${sales_per_unit_scenario:.2f}

Revenue Per Unit Change:
{sales_per_unit_change:.1f}%

Profit Margin Baseline:
{profit_margin_baseline:.1f}%

Profit Margin Scenario:
{profit_margin_scenario:.1f}%

Profit Margin Change:
{profit_margin_change:.1f}%

Growth Efficiency:
{growth_efficiency:.2f}
"""

if prompt:
    analysis_context = build_analysis_context()
    intent = detect_intent(prompt)

    context = json.dumps(
        st.session_state.get(
            "forecast_results",
            {}
        ),
        indent=2
    )

    # ----------------------------------
    # HIGHEST SALES MONTH
    # ----------------------------------

    if intent == "highest_sales":

        result = highest_sales_month(
            st.session_state[
                "monthly_forecast_df"
            ]
        )

        response = ask_llm(
            f"""
            Highest Forecasted Sales Month

            Month:
            {result['month']}

            Sales:
            ${result['sales']:,.2f}

            Explain the business significance
            and possible actions management
            should take.
            """,
            analysis_context
        )

    # ----------------------------------
    # LOWEST SALES MONTH
    # ----------------------------------

    elif intent == "lowest_sales":

        result = lowest_sales_month(
            st.session_state[
                "monthly_forecast_df"
            ]
        )

        response = ask_llm(
            f"""
            Lowest Forecasted Sales Month

            Month:
            {result['month']}

            Sales:
            ${result['sales']:,.2f}

            Explain why this could happen
            and recommend actions.
            """,
            analysis_context
        )

    # ----------------------------------
    # BEST PROFIT MONTH
    # ----------------------------------

    elif intent == "best_profit":

        result = best_profit_month(
            st.session_state[
                "monthly_forecast_df"
            ]
        )

        response = ask_llm(
            f"""
            Best Forecasted Profit Month

            Month:
            {result['month']}

            Profit:
            ${result['profit']:,.2f}

            Explain why profitability
            is strongest here.
            """,
            analysis_context
        )

    # ----------------------------------
    # GROWTH RATE
    # ----------------------------------

    elif intent == "growth":

        growth = sales_growth_rate(
            st.session_state[
                "monthly_forecast_df"
            ]
        )

        response = ask_llm(
            f"""
            Forecasted Sales Growth Rate

            {growth:.2f}%

            Explain what this means
            for the business.
            """,
            analysis_context
        )

    # ----------------------------------
    # BEST YEAR
    # ----------------------------------

    elif intent == "best_year":

        result = top_forecast_year(
            st.session_state[
                "yearly_summary_df"
            ]
        )

        response = ask_llm(
            f"""
            Best Forecasted Year

            Year:
            {result['year']}

            Sales:
            ${result['sales']:,.2f}

            Explain why this year
            performs best.
            """,
            analysis_context
        )

    # ----------------------------------
    # WEB SEARCH
    # ----------------------------------

    elif intent == "web":

        search_results = web_search(
            prompt
        )

        response = ask_llm(
            f"""
            Forecast Context:

            {context}

            Web Search Results:

            {search_results}

            User Question:

            {prompt}
            """,
            analysis_context
        )

    # ----------------------------------
    # DEFAULT CHAT
    # ----------------------------------

    else:

        response = ask_llm(
            f"""
            Using ONLY the forecast metrics provided.

            Do not assume causes that are not present
            in the data.

            Explain:

            1. What improved
            2. What worsened
            3. Key business risks
            4. Key opportunities
            5. Recommended management actions

            User Question:
            {prompt}

            Use actual forecast values and percentages.
            """,
            analysis_context
        )

    # ----------------------------------
    # SAVE CHAT HISTORY
    # ----------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    st.rerun()