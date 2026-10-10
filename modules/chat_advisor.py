"""
Chat Advisor Module (Groq-powered)
-------------------------------------
A genuinely conversational financial advisor: the user can ask open-ended
questions in plain English ("where should I cut back?", "is now a good
time to buy a laptop?") and get a real, reasoned answer grounded in their
actual spending data -- not a canned template.

Uses Groq's free API tier (https://console.groq.com) via their official
Python SDK. No credit card required to get a key. If GROQ_API_KEY isn't
set, or the `groq` package isn't installed, or the call fails for any
reason (no internet, rate limit, etc.), the caller should fall back to
the rule-based `advisor.financial_health_report` -- this module never
crashes the app, it just returns None when it can't help.
"""

import os

GROQ_MODEL = "openai/gpt-oss-120b"  # Groq's free-tier flagship model as of 2026


def groq_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def _build_context(df, alerts, anomalies, recurring, budgets) -> str:
    """Summarize the user's real financial data into a compact text block
    the model can reason over -- this is what grounds the answers in fact
    instead of letting the model guess or hallucinate numbers."""
    income = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].abs().sum()
    cat_spend = (
        df[df["amount"] < 0].groupby("category")["amount"]
        .apply(lambda s: s.abs().sum()).sort_values(ascending=False)
    )

    lines = [
        f"Total income this period: ₹{income:,.0f}",
        f"Total expenses this period: ₹{expenses:,.0f}",
        f"Net cash flow: ₹{income - expenses:,.0f}",
        "",
        "Spending by category (with budget, if set):",
    ]
    for cat, spent in cat_spend.items():
        budget = budgets.get(cat)
        budget_str = f", budget ₹{budget:,.0f}" if budget else ""
        lines.append(f"  - {cat}: ₹{spent:,.0f}{budget_str}")

    if alerts:
        lines.append("\nBudget alerts:")
        for a in alerts:
            lines.append(f"  - {a['message']}")

    if len(anomalies):
        lines.append("\nUnusually large transactions detected:")
        for _, row in anomalies.head(5).iterrows():
            lines.append(f"  - {row['description']}: ₹{row['abs_amount']:,.0f}")

    if len(recurring):
        lines.append(f"\nRecurring subscriptions: {len(recurring)} totalling "
                      f"₹{recurring['abs_amount'].sum():,.0f}/cycle")

    return "\n".join(lines)


def ask(question: str, df, alerts, anomalies, recurring, budgets: dict,
        history: list = None) -> str:
    """Ask the Groq-hosted LLM a free-form question about the user's
    finances, grounded in their real data. Returns the answer text, or
    raises an exception on failure (the caller should catch it and fall
    back to the rule-based advisor)."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    context = _build_context(df, alerts, anomalies, recurring, budgets)

    system_prompt = (
        "You are a helpful, concise personal finance advisor embedded in a "
        "budgeting app. Answer the user's question using ONLY the financial "
        "data below -- do not invent numbers. Give practical, specific "
        "advice (what to cut, when it's safe to spend, how to save more). "
        "Keep answers to 2-4 short sentences unless the user asks for more "
        "detail. Use ₹ for currency.\n\n"
        f"--- User's current financial data ---\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-6:])  # keep last few turns for context, bounded
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()
