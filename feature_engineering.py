import pandas as pd
import numpy as np

def validate_and_clean_data(df):

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    required_cols = [
        "order_date",
        "state",
        "sub_category",
        "category",
        "sales",
        "quantity",
        "profit",
        "discount"
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )
    
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        format="mixed",
        errors="coerce"
    )

    if df["order_date"].isna().any():

        bad_rows = df[
            df["order_date"].isna()
        ]

        raise ValueError(
            f"Invalid dates found:\n{bad_rows.head()}"
        )


    months_of_history = (
        df['order_date'].dt.to_period('M').nunique()
    )

    if months_of_history < 12:
        raise ValueError(
            "At least 12 months of historical data is required."
        )

    return df

def engineer_features(df):
    df = validate_and_clean_data(df)
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['day'] = df['order_date'].dt.day
    df['dayofweek'] = df['order_date'].dt.dayofweek
    df['quarter'] = df['order_date'].dt.quarter
    drop_cols =[
    'row_id', 'order_id', 'customer_id',
    'customer_name', 'product_id',
    'ship_date', 'ship_mode', 
    'postal_code', 'region'
    ]

    df = df.drop(columns=drop_cols, errors='ignore')
    monthly_sales = (
        df.groupby([
            pd.Grouper(key='order_date', freq='ME'),
            'state',
            'category',
            'sub_category'
        ])
        .agg({
            'sales': 'sum',
            'quantity': 'sum',
            'profit': 'sum',
            'discount': 'mean' 
        })
        .reset_index()
    )
    order_counts = (
        df.groupby([
            pd.Grouper(key='order_date', freq='ME'),
            'state',
            'sub_category'
        ])
        .size()
        .reset_index(name='order_count')
    )
    monthly_sales = monthly_sales.merge(
        order_counts,
        on=['order_date', 'state', 'sub_category'],
        how='left'
    )
    monthly_sales['price'] = monthly_sales['sales'] / (monthly_sales['quantity'] + 1e-5)
    monthly_sales['profit_margin'] = monthly_sales['profit'] / (monthly_sales['sales'] + 1e-5)
    def is_peak_season(date):
        month = date.month
        
        if month in [11, 12]:
            return 1
        
        if month in [6, 7]:
            return 1
        
        return 0

    monthly_sales['is_peak_season'] = monthly_sales['order_date'].apply(is_peak_season)
    def holiday_weight(date):
        month = date.month
        
        if month == 12: 
            return 3
        elif month == 11:  
            return 2
        elif month == 1:  
            return 1
        else:
            return 0

    monthly_sales['holiday_impact'] = monthly_sales['order_date'].apply(holiday_weight)
    monthly_sales['days_to_christmas'] = (
        pd.to_datetime(monthly_sales['order_date'].dt.year.astype(str) + '-12-25')
        - monthly_sales['order_date']
    ).dt.days

    monthly_sales = monthly_sales.sort_values(
        ['state', 'category', 'sub_category', 'order_date']
    )

    monthly_sales['order_lag_1'] = (
        monthly_sales
        .groupby(
            ['state', 'category', 'sub_category']
        )['order_count']
        .shift(1)
    )

    monthly_sales['order_rolling_3'] = (
        monthly_sales
        .groupby(
            ['state', 'category', 'sub_category']
        )['order_count']
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    monthly_sales['lag_1'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['sales']
        .shift(1)
    )

    monthly_sales['lag_2'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['sales']
        .shift(2)
    )

    monthly_sales['lag_12'] = (
        monthly_sales
        .groupby(['state','sub_category'])['sales']
        .shift(12)
    )

    monthly_sales['rolling_3'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['sales']
        .transform(lambda x: x.shift(1).rolling(3).mean())
    )

    monthly_sales['rolling_6'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['sales']
        .transform(lambda x: x.shift(1).rolling(6).mean())
    )

    monthly_sales['rolling_12'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['sales']
        .transform(lambda x: x.shift(1).rolling(12).mean())
    )

    monthly_sales['quantity_lag_1'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['quantity']
        .shift(1)
    )

    monthly_sales['quantity_lag_2'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['quantity']
        .shift(2)
    )

    monthly_sales['quantity_rolling_3'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['quantity']
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    monthly_sales['profit_lag_1'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['profit']
        .shift(1)
    )

    monthly_sales['profit_lag_2'] = (
        monthly_sales
        .groupby(['state','sub_category'])['profit']
        .shift(2)
    )

    monthly_sales['discount_x_sales'] = (
        monthly_sales['discount'] *
        monthly_sales['sales']
    )

    monthly_sales['quantity_x_price'] = (
        monthly_sales['quantity'] *
        monthly_sales['price']
    )

    monthly_sales['profit_rolling_3'] = (
        monthly_sales
        .groupby(['state','sub_category'])['profit']
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    monthly_sales['discount_rolling_3'] = (
        monthly_sales
        .groupby(['state','sub_category'])['discount']
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    monthly_sales['profit_momentum_1'] = (
        monthly_sales['profit_lag_1']
        /
        (monthly_sales['profit_lag_2'] + 1e-5)
    )

    monthly_sales['discount_lag_1'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['discount']
        .shift(1)
    )

    monthly_sales['black_friday_flag'] = 0

    monthly_sales.loc[
        monthly_sales['order_date'].dt.month == 11,
        'black_friday_flag'
    ] = 1

    monthly_sales['profit_margin_lag_1'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['profit_margin']
        .shift(1)
    )

    monthly_sales['profit_margin_lag_2'] = (
        monthly_sales
        .groupby(['state', 'sub_category'])['profit_margin']
        .shift(2)
    )
    monthly_sales['order_count_lag_1'] = (
        monthly_sales
        .groupby(
            ['state', 'category', 'sub_category']
        )['order_count']
        .shift(1)
    )

    monthly_sales['order_count_lag_2'] = (
        monthly_sales
        .groupby(
            ['state', 'category', 'sub_category']
        )['order_count']
        .shift(2)
    )

    monthly_sales['order_count_rolling_3'] = (
        monthly_sales
        .groupby(
            ['state', 'category', 'sub_category']
        )['order_count']
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    monthly_sales['quarter'] = (
        monthly_sales['order_date'].dt.quarter
    )

    monthly_sales['year'] = (
        monthly_sales['order_date'].dt.year
    )

    monthly_sales['time_index'] = np.arange(len(monthly_sales))
    monthly_sales['month'] = monthly_sales['order_date'].dt.month
    monthly_sales['month_sin'] = np.sin(2 * np.pi * monthly_sales['month']/12)
    monthly_sales['month_cos'] = np.cos(2 * np.pi * monthly_sales['month']/12)

    monthly_sales['momentum_1'] = (
        monthly_sales['lag_1'] - monthly_sales['lag_2']
    )

    monthly_sales = monthly_sales.dropna()

    return monthly_sales

def build_future_features(
    future_df,
    historical_df
):
    """
    Populate future forecast periods with the latest
    historical feature values so the models receive
    all required lag/rolling features.
    """

    if historical_df.empty:
        raise ValueError(
            "No historical data available for this selection."
        )

    latest_row = (
        historical_df
        .sort_values("order_date")
        .iloc[-1]
    )

    # --------------------------------------------------
    # SALES FEATURES
    # --------------------------------------------------

    future_df["lag_1"] = latest_row["lag_1"]
    future_df["lag_2"] = latest_row["lag_2"]
    future_df["lag_12"] = latest_row["lag_12"]

    future_df["rolling_3"] = latest_row["rolling_3"]
    future_df["rolling_6"] = latest_row["rolling_6"]
    future_df["rolling_12"] = latest_row["rolling_12"]

    future_df["momentum_1"] = latest_row["momentum_1"]

    # --------------------------------------------------
    # QUANTITY FEATURES
    # --------------------------------------------------

    future_df["quantity_lag_1"] = latest_row["quantity_lag_1"]
    future_df["quantity_lag_2"] = latest_row["quantity_lag_2"]
    future_df["quantity_rolling_3"] = latest_row["quantity_rolling_3"]

    # --------------------------------------------------
    # PROFIT FEATURES
    # --------------------------------------------------

    future_df["profit_lag_1"] = latest_row["profit_lag_1"]
    future_df["profit_lag_2"] = latest_row["profit_lag_2"]

    future_df["profit_rolling_3"] = (
        latest_row["profit_rolling_3"]
    )

    future_df["profit_momentum_1"] = (
        latest_row["profit_momentum_1"]
    )

    # --------------------------------------------------
    # DISCOUNT FEATURES
    # --------------------------------------------------

    future_df["discount_lag_1"] = (
        latest_row["discount_lag_1"]
    )

    future_df["discount_rolling_3"] = (
        latest_row["discount_rolling_3"]
    )

    # --------------------------------------------------
    # ORDER FEATURES
    # --------------------------------------------------

    future_df["order_count"] = (
        latest_row["order_count"]
    )

    future_df["order_lag_1"] = (
        latest_row["order_lag_1"]
    )

    future_df["order_rolling_3"] = (
        latest_row["order_rolling_3"]
    )

    future_df["order_count_lag_1"] = (
        latest_row["order_count_lag_1"]
    )

    future_df["order_count_lag_2"] = (
        latest_row["order_count_lag_2"]
    )

    future_df["order_count_rolling_3"] = (
        latest_row["order_count_rolling_3"]
    )

    # --------------------------------------------------
    # PRICE / MARGIN FEATURES
    # --------------------------------------------------

    future_df["price"] = latest_row["price"]

    future_df["profit_margin"] = (
        latest_row["profit_margin"]
    )

    future_df["profit_margin_lag_1"] = (
        latest_row["profit_margin_lag_1"]
    )

    future_df["profit_margin_lag_2"] = (
        latest_row["profit_margin_lag_2"]
    )

    # --------------------------------------------------
    # CURRENT VALUES
    # --------------------------------------------------

    future_df["quantity"] = (
        latest_row["quantity"]
    )

    future_df["sales"] = (
        latest_row["sales"]
    )

    future_df["profit"] = (
        latest_row["profit"]
    )

    future_df["discount"] = (
        latest_row["discount"]
    )

    # --------------------------------------------------
    # SEASONAL FEATURES
    # --------------------------------------------------

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
        - future_df["order_date"]
    ).dt.days

    # --------------------------------------------------
    # INTERACTION FEATURES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # TIME INDEX
    # --------------------------------------------------

    future_df["time_index"] = (
        np.arange(len(future_df))
        +
        latest_row["time_index"]
        +
        1
    )

    return future_df
