from openai import OpenAI
import streamlit as st

client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

SYSTEM_PROMPT = """
You are a senior business intelligence consultant.

Rules:

1. Use ONLY the forecast data provided.
2. Never invent causes that are not present in the data.
3. If the data does not explain WHY something happened,
   say additional business data is required.
4. Focus on:
   - sales
   - profit
   - quantity
   - forecast impact
   - business decisions
5. Use actual values provided.
6. Use percentages provided.
7. Keep recommendations evidence-based.
8. Do not assume seasonality unless shown in the data.
9. Explain findings in plain English.
10. Give actionable recommendations.
11. Do not perform calculations.
12. Use the supplied KPI summary.
"""


def ask_llm(
    user_prompt,
    context=""
):

    response = client.chat.completions.create(

        model="openrouter/free",

        temperature=0.1,

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": f"""
Forecast KPI Summary:

{context}

Question:

{user_prompt}
"""
            }

        ]
    )

    return response.choices[0].message.content