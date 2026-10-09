def detect_intent(prompt):

    prompt = prompt.lower()

    if "highest sales" in prompt:
        return "highest_sales"

    if "best sales month" in prompt:
        return "highest_sales"

    if "lowest sales" in prompt:
        return "lowest_sales"

    if "best profit" in prompt:
        return "best_profit"

    if "worst profit" in prompt:
        return "worst_profit"

    if "average sales" in prompt:
        return "avg_sales"

    if "growth rate" in prompt:
        return "growth"

    if "best year" in prompt:
        return "best_year"

    if "forecast" in prompt:
        return "forecast"

    if "what if" in prompt:
        return "scenario"

    if "search" in prompt:
        return "web"

    return "chat"