import pandas as pd


def highest_sales_month(df):

    row = df.loc[
        df["Baseline Sales"].idxmax()
    ]

    return {
        "month": row["Month"],
        "sales": float(
            row["Baseline Sales"]
        )
    }


def lowest_sales_month(df):

    row = df.loc[
        df["Baseline Sales"].idxmin()
    ]

    return {
        "month": row["Month"],
        "sales": float(
            row["Baseline Sales"]
        )
    }


def best_profit_month(df):

    row = df.loc[
        df["Baseline Profit"].idxmax()
    ]

    return {
        "month": row["Month"],
        "profit": float(
            row["Baseline Profit"]
        )
    }


def worst_profit_month(df):

    row = df.loc[
        df["Baseline Profit"].idxmin()
    ]

    return {
        "month": row["Month"],
        "profit": float(
            row["Baseline Profit"]
        )
    }


def average_monthly_sales(df):

    return float(
        df["Baseline Sales"].mean()
    )


def total_forecast_sales(df):

    return float(
        df["Baseline Sales"].sum()
    )


def total_forecast_profit(df):

    return float(
        df["Baseline Profit"].sum()
    )


def top_forecast_year(yearly_df):

    row = yearly_df.loc[
        yearly_df["Baseline Sales"].idxmax()
    ]

    return {
        "year": int(row["Year"]),
        "sales": float(
            row["Baseline Sales"]
        )
    }


def sales_growth_rate(df):

    first = df.iloc[0]["Baseline Sales"]

    last = df.iloc[-1]["Baseline Sales"]

    growth = (
        (last - first)
        / first
    ) * 100

    return float(growth)