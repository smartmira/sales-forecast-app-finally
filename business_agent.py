import pandas as pd


REQUIRED_COLUMNS = [
    "order_date",
    "state",
    "category",
    "sub_category",
    "sales",
    "quantity",
    "profit",
    "discount"
]


def validate_business_data(df):

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    return missing


def generate_business_summary(df):

    summary = f"""
Business Dataset Loaded Successfully

Rows:
{len(df):,}

States:
{df['state'].nunique()}

Categories:
{df['category'].nunique()}

Sub Categories:
{df['sub_category'].nunique()}

Date Range:
{df['order_date'].min()}
to
{df['order_date'].max()}

Total Sales:
${df['sales'].sum():,.0f}

Total Profit:
${df['profit'].sum():,.0f}

Total Quantity:
{df['quantity'].sum():,.0f}
"""

    return summary