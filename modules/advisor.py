"""
Advisor Module
---------------
Synthesizes the outputs of the other modules into a natural-language
executive summary and actionable recommendations. Template-based by
default (fast, deterministic, offline); can be swapped for an LLM call
the same way classifier.py is, if richer prose is wanted.

Also includes a rule-based "financial health" engine that answers the
practical where/when/how-much-to-spend-or-save questions a user cares
about, grounded entirely in their own transaction data and budgets.
"""

# Rough guideline shares of monthly spend, used only to flag categories
# that are unusually large relative to typical personal-finance advice
# (e.g. the "50/30/20 rule" style guidance) -- NOT a hard rule, just a
# heuristic trigger for a suggestion.
_ESSENTIAL_CATEGORIES = {"Rent/Housing", "Groceries", "Utilities", "Healthcare"}
_DISCRETIONARY_CATEGORIES = {"Food & Dining", "Shopping", "Entertainment", "Subscriptions"}


def generate_summary(df, alerts, anomalies, recurring) -> str:
    income = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].abs().sum()
    net = income - expenses
    top_categories = (
        df[df["amount"] < 0].groupby("category")["amount"]
        .apply(lambda s: s.abs().sum())
        .sort_values(ascending=False)
        .head(3)
    )

    lines = []
    lines.append(f"Total income: ₹{income:,.0f} | Total expenses: ₹{expenses:,.0f} | "
                  f"Net cash flow: ₹{net:,.0f}")
    lines.append("Top spending categories: " +
                  ", ".join(f"{c} (₹{v:,.0f})" for c, v in top_categories.items()))

    if alerts:
        lines.append(f"⚠ {len(alerts)} budget alert(s): " +
                      "; ".join(a["message"] for a in alerts[:3]))
    else:
        lines.append("✓ All categories are within their monthly budget.")

    if len(anomalies):
        lines.append(f"⚠ {len(anomalies)} unusually large transaction(s) detected "
                      f"(largest: {anomalies.iloc[0]['description']} — "
                      f"₹{anomalies.iloc[0]['abs_amount']:,.0f}).")
    else:
        lines.append("✓ No abnormal spending spikes detected.")

    if len(recurring):
        lines.append(f"↻ {len(recurring)} likely recurring subscription(s) found, "
                      f"totalling ₹{recurring['abs_amount'].sum():,.0f}/cycle — "
                      "review for ones you no longer use.")

    if net < 0:
        lines.append("Recommendation: expenses exceeded income this period — "
                      "consider trimming the top spending category above.")
    elif top_categories.iloc[0] > 0.4 * expenses:
        lines.append(f"Recommendation: {top_categories.index[0]} alone accounts for "
                      f"over 40% of spending — worth a closer look.")
    else:
        lines.append("Recommendation: spending is fairly balanced across categories — "
                      "keep monitoring for budget alerts.")

    return "\n".join(lines)


def financial_health_report(df, alerts, anomalies, recurring, budgets: dict) -> dict:
    """Rule-based answers to: how much am I saving, where should I cut back,
    and is now a good time to spend? Returns a dict of labelled sections so
    the UI can render them as cards rather than one big paragraph."""
    income = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].abs().sum()
    net = income - expenses
    savings_rate = (net / income * 100) if income > 0 else 0

    cat_spend = (
        df[df["amount"] < 0].groupby("category")["amount"]
        .apply(lambda s: s.abs().sum())
    )

    # --- Where to spend / cut back ---
    cuts = []
    for cat, spent in cat_spend.sort_values(ascending=False).items():
        limit = budgets.get(cat)
        if limit and spent > limit:
            cuts.append(f"**{cat}**: ₹{spent:,.0f} spent vs ₹{limit:,.0f} budget — "
                        f"over by ₹{spent - limit:,.0f}. This is the first place to cut.")
        elif cat in _DISCRETIONARY_CATEGORIES and expenses > 0 and spent > 0.25 * expenses:
            cuts.append(f"**{cat}** is discretionary and makes up "
                        f"{spent / expenses * 100:.0f}% of your spending — "
                        f"trimming this won't affect essentials.")
    if not cuts:
        cuts.append("No category stands out as overspent — spending looks reasonably balanced.")

    # --- Where it's safe to keep spending ---
    safe = []
    for cat, spent in cat_spend.items():
        limit = budgets.get(cat)
        if limit and spent < 0.7 * limit:
            headroom = limit - spent
            safe.append(f"**{cat}**: ₹{headroom:,.0f} of budget still unused this period.")
    if not safe:
        safe.append("Most categories are close to or over budget — limited headroom anywhere right now.")

    # --- When to spend (timing guidance) ---
    timing = []
    if savings_rate < 0:
        timing.append("You're spending more than you earn this period — "
                       "hold off on non-essential purchases until income catches up.")
    elif savings_rate < 10:
        timing.append(f"Savings rate is only {savings_rate:.0f}% — fine for small purchases, "
                       "but delay big-ticket items until a stronger month.")
    elif savings_rate < 25:
        timing.append(f"Savings rate is a healthy {savings_rate:.0f}% — a reasonable "
                       "time for planned, moderate purchases.")
    else:
        timing.append(f"Savings rate is strong at {savings_rate:.0f}% — a good window "
                       "for a bigger planned purchase or extra savings/investment.")

    if len(anomalies):
        timing.append("Note: an unusually large transaction was detected this period — "
                       "factor that into any near-term spending decisions.")

    # --- Save more, concretely ---
    save_more = []
    if len(recurring):
        save_more.append(f"Cancel unused subscriptions: {len(recurring)} recurring charge(s) "
                         f"total ₹{recurring['abs_amount'].sum():,.0f}/cycle.")
    top_discretionary = cat_spend[cat_spend.index.isin(_DISCRETIONARY_CATEGORIES)]
    if len(top_discretionary):
        biggest = top_discretionary.idxmax()
        save_more.append(f"Cutting {biggest} spending by 20% would save roughly "
                         f"₹{top_discretionary.max() * 0.2:,.0f}/period.")
    if not save_more:
        save_more.append("No obvious quick wins found — spending is already fairly lean.")

    return {
        "savings_rate": savings_rate,
        "net": net,
        "where_to_cut": cuts,
        "where_safe_to_spend": safe,
        "when_to_spend": timing,
        "how_to_save_more": save_more,
    }
