import pandas as pd
import numpy as np


# --------------------------------------------------
# FUTURE CALENDAR
# --------------------------------------------------

def create_future_calendar(
    start_date,
    forecast_months
):

    future_dates = pd.date_range(
        start=start_date,
        periods=forecast_months,
        freq="M"
    )

    future_df = pd.DataFrame({
        "order_date": future_dates
    })

    future_df["month"] = (
        future_df["order_date"].dt.month
    )

    future_df["quarter"] = (
        future_df["order_date"].dt.quarter
    )

    future_df["year"] = (
        future_df["order_date"].dt.year
    )

    future_df["month_sin"] = np.sin(
        2 * np.pi *
        future_df["month"] / 12
    )

    future_df["month_cos"] = np.cos(
        2 * np.pi *
        future_df["month"] / 12
    )

    future_df["is_peak_season"] = (
        future_df["month"]
        .isin([6, 7, 11, 12])
        .astype(int)
    )

    future_df["days_to_christmas"] = (
        pd.to_datetime(
            future_df["year"].astype(str)
            + "-12-25"
        )
        -
        future_df["order_date"]
    ).dt.days

    return future_df


# --------------------------------------------------
# HOLIDAY WEIGHT
# --------------------------------------------------

def holiday_weight(month):

    if month == 12:
        return 3

    elif month == 11:
        return 2

    elif month == 1:
        return 1

    return 0


# --------------------------------------------------
# SCENARIO FEATURES
# --------------------------------------------------

def add_scenario_features(
    future_df,
    scenario
):

    future_df["holiday_impact"] = (
        future_df["month"]
        .apply(holiday_weight)
    )

    future_df["black_friday_flag"] = (
        future_df["month"] == 11
    ).astype(int)

    future_df["promotion_flag"] = (
        scenario["promotion_flag"]
    )

    future_df["economic_outlook"] = (
        scenario["economic_outlook"]
    )

    future_df["demand_growth_pct"] = (
        scenario["demand_growth_pct"]
    )

    future_df["price_growth_pct"] = (
        scenario["price_growth_pct"]
    )

    future_df["discount_pct"] = (
        scenario["discount_pct"]
    )

    return future_df


# --------------------------------------------------
# QUANTITY FORECAST
# --------------------------------------------------

def forecast_quantity(
    future_df,
    quantity_model,
    quantity_features
):

    X = future_df[
        quantity_features
    ]

    future_df["quantity"] = (
        quantity_model.predict(X)
    )

    return future_df


# --------------------------------------------------
# APPLY BUSINESS SCENARIO
# --------------------------------------------------

def apply_scenario(
    future_df,
    scenario,
    latest_price
):

    # Demand increase
    future_df["quantity"] *= (
        1 +
        scenario["demand_growth_pct"] / 100
    )

    # Promotion effect
    if scenario["promotion_flag"] == 1:
        future_df["quantity"] *= 1.10

    # Price increase
    future_df["price"] = (
        latest_price *
        (
            1 +
            scenario["price_growth_pct"] / 100
        )
    )

    # Discount
    future_df["discount"] = (
        scenario["discount_pct"] / 100
    )

    # Economic outlook effect
    if scenario["economic_outlook"] == "positive":
        future_df["quantity"] *= 1.05

    elif scenario["economic_outlook"] == "negative":
        future_df["quantity"] *= 0.95

    return future_df


# --------------------------------------------------
# SALES FORECAST
# --------------------------------------------------

def forecast_sales(
    future_df,
    sales_model,
    sales_features
):

    X = future_df[
        sales_features
    ]

    future_df["sales"] = (
        sales_model.predict(X)
    )

    return future_df


# --------------------------------------------------
# PROFIT FORECAST
# --------------------------------------------------

def forecast_profit(
    future_df,
    profit_model,
    profit_features
):

    future_df["discount_x_sales"] = (
        future_df["discount"]
        *
        future_df["sales"]
    )

    future_df["quantity_x_price"] = (
        future_df["quantity"]
        *
        future_df["price"]
    )

    X = future_df[
        profit_features
    ]

    future_df["profit"] = (
        profit_model.predict(X)
    )

    return future_df