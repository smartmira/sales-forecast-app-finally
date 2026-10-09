import pandas as pd
import numpy as np

np.random.seed(42)

states = [
    "Texas",
    "Florida",
    "Nevada",
    "Colorado",
    "Georgia"
]

products = {
    "Electronics": [
        "Tablets",
        "Gaming Consoles",
        "Smart Watches"
    ],
    "Home Goods": [
        "Air Fryers",
        "Coffee Machines",
        "Vacuum Cleaners"
    ],
    "Sports": [
        "Treadmills",
        "Exercise Bikes",
        "Dumbbells"
    ]
}

dates = pd.date_range(
    start="2022-01-01",
    end="2024-12-31",
    freq="D"
)

rows = []

for date in dates:

    for _ in range(
        np.random.randint(3, 10)
    ):

        state = np.random.choice(states)

        category = np.random.choice(
            list(products.keys())
        )

        sub_category = np.random.choice(
            products[category]
        )

        quantity = np.random.randint(
            1,
            15
        )

        base_price = np.random.uniform(
            50,
            2000
        )

        discount = np.random.choice(
            [0, 0.05, 0.10, 0.15, 0.20]
        )

        sales = (
            quantity *
            base_price *
            (1 - discount)
        )

        profit = (
            sales *
            np.random.uniform(
                0.08,
                0.25
            )
        )

        rows.append([
            date,
            state,
            category,
            sub_category,
            sales,
            quantity,
            profit,
            discount
        ])

df = pd.DataFrame(
    rows,
    columns=[
        "order_date",
        "state",
        "category",
        "sub_category",
        "sales",
        "quantity",
        "profit",
        "discount"
    ]
)

#df.to_csv(
    #"synthetic_sales_data.csv",
    #index=False
#)

print(df.shape)
print("saved")
print(df['order_date'].head(10))