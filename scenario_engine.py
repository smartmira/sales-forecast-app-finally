# scenario_engine.py

import pandas as pd


def apply_business_scenario(
    baseline_df,
    scenario
):

    scenario_df = baseline_df.copy()

    demand_factor = (
        1 +
        scenario["demand_growth_pct"] / 100
    )

    price_factor = (
        1 +
        scenario["price_growth_pct"] / 100
    )

    discount_factor = (
        1 -
        scenario["discount_pct"] / 100
    )

    scenario_df["quantity"] = (
        scenario_df["quantity"]
        * demand_factor
    )

    scenario_df["sales"] = (
        scenario_df["sales"]
        * demand_factor
        * price_factor
        * discount_factor
    )

    if scenario["promotion_flag"] == 1:

        scenario_df["sales"] *= 1.10

        scenario_df["quantity"] *= 1.15

    if scenario["economic_outlook"] == "positive":

        scenario_df["sales"] *= 1.10

    elif scenario["economic_outlook"] == "negative":

        scenario_df["sales"] *= 0.90

    profit_margin = (
        baseline_df["profit"].sum()
        /
        max(baseline_df["sales"].sum(), 1)
    )

    scenario_df["profit"] = (
        scenario_df["sales"]
        * profit_margin
    )

    return scenario_df