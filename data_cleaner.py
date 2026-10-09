# data_cleaner.py

import pandas as pd


def clean_dates(df):

    if "order_date" in df.columns:

        df["order_date"] = pd.to_datetime(
            df["order_date"],
            errors="coerce"
        )

    return df


def clean_sales(df):

    if "sales" in df.columns:

        df["sales"] = (
            df["sales"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        df["sales"] = pd.to_numeric(
            df["sales"],
            errors="coerce"
        )

    return df


def clean_quantity(df):

    if "quantity" in df.columns:

        df["quantity"] = pd.to_numeric(
            df["quantity"],
            errors="coerce"
        )

    return df


def clean_profit(df):

    if "profit" in df.columns:

        df["profit"] = (
            df["profit"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        df["profit"] = pd.to_numeric(
            df["profit"],
            errors="coerce"
        )

    return df


def clean_discount(df):

    if "discount" in df.columns:

        df["discount"] = (
            df["discount"]
            .astype(str)
            .str.replace("%", "", regex=False)
        )

        df["discount"] = pd.to_numeric(
            df["discount"],
            errors="coerce"
        )

        df["discount"] = (
            df["discount"] / 100
        )

    return df


def clean_dataset(df):

    df = clean_dates(df)

    df = clean_sales(df)

    df = clean_quantity(df)

    df = clean_profit(df)

    df = clean_discount(df)

    return df