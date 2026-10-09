import pandas as pd
from forecast_engine import (
    create_future_calendar,
    add_scenario_features,
    forecast_quantity,
    forecast_sales,
    forecast_profit
)

from feature_engineering import (
    build_future_features
)


def run_forecast(
    business_df,
    state,
    category,
    sub_category,
    scenario,
    quantity_model,
    sales_model,
    profit_model,
    quantity_features,
    sales_features,
    profit_features,
    forecast_months
):

    forecast_start = (
        business_df["order_date"].max()
        + pd.offsets.MonthEnd(1)
    )

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

    return future_df