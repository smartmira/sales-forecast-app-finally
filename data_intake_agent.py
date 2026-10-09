# data_intake_agent.py

from chatbot import ask_llm
import json


EXPECTED_SCHEMA = [
    "order_date",
    "state",
    "category",
    "sub_category",
    "sales",
    "quantity",
    "profit",
    "discount"
]


def auto_rule_mapping(df):

    rename_map = {}

    for col in df.columns:

        c = col.lower().strip()

        if "date" in c:
            rename_map[col] = "order_date"

        elif "revenue" in c or "sales" in c:
            rename_map[col] = "sales"

        elif "quantity" in c or "qty" in c or "unit" in c:
            rename_map[col] = "quantity"

        elif "profit" in c or "margin" in c:
            rename_map[col] = "profit"

        elif "discount" in c:
            rename_map[col] = "discount"

        elif "region" in c or "state" in c or "location" in c:
            rename_map[col] = "state"

        elif (
            ("sub" in c and "category" in c)
            or
            ("product" in c and "type" in c)
        ):
            rename_map[col] = "sub_category"

        elif (
            "category" in c
            or
            "family" in c
        ):
            rename_map[col] = "category"

        mapped_targets = list(rename_map.values())

        duplicates = [
            col
            for col in mapped_targets
            if mapped_targets.count(col) > 1
        ]

        if duplicates:
            print(
                "WARNING duplicate targets:",
                duplicates
            )

    return df.rename(columns=rename_map)


def find_unmapped_columns(df):

    mapped_columns = set(df.columns)

    unmapped = []

    for col in mapped_columns:

        if col not in EXPECTED_SCHEMA:
            unmapped.append(col)

    return unmapped


def qwen_column_mapper(columns):

    prompt = f"""
You are a data mapping assistant.

Map uploaded columns to:

order_date
state
category
sub_category
sales
quantity
profit
discount

Columns:

{columns}

Return ONLY valid JSON.

Example:

{{
"Revenue":"sales",
"Units":"quantity"
}}
"""

    response = ask_llm(prompt)

    try:
        mapping = json.loads(response)
        return mapping

    except:
        return {}


def standardize_dataset(df):

    print("Columns before mapping:")
    print(df.columns.tolist())

    df = auto_rule_mapping(df)

    print("Columns after mapping:")
    print(df.columns.tolist())

    unmapped = find_unmapped_columns(df)

    if len(unmapped) > 0:

        ai_mapping = qwen_column_mapper(
            unmapped
        )

        df = df.rename(
            columns=ai_mapping
        )

    return df