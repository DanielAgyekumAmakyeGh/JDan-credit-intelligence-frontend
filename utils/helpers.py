def format_currency(amount):
    if amount is None:
        return "GHS 0"
    return f"GHS {amount:,.2f}"
