"""
AI Classification Module
--------------------------
Maps unstructured transaction descriptions into structured spending
categories.

Two engines:
  1. LLM zero-shot classifier (Anthropic Claude) - used automatically
     when ANTHROPIC_API_KEY is set in the environment. Requires network
     access and the `anthropic` package.
  2. Rule-based keyword classifier - always available, used as the
     offline fallback and for fast batch processing without API cost.

Both return one of CATEGORIES for a given transaction description.
"""

import os
import re
import json

CATEGORIES = [
    "Income", "Rent/Housing", "Groceries", "Food & Dining", "Transportation",
    "Utilities", "Subscriptions", "Shopping", "Entertainment",
    "Healthcare", "Miscellaneous",
]

# Keyword rules used by the offline fallback engine. Order matters:
# first match wins.
_RULES = [
    ("Income", r"salary|payroll|stipend|refund|interest credit|credited by"),
    ("Rent/Housing", r"\brent\b|landlord|mortgage|housing society"),
    ("Groceries", r"grocery|supermarket|bigbasket|d ?mart|reliance fresh|zepty|zepto|blinkit"),
    ("Food & Dining", r"restaurant|cafe|coffee|zomato|swiggy|food|dine|pizza|starbucks|mcdonald"),
    ("Transportation", r"uber|ola|fuel|petrol|diesel|metro|irctc|flight|indigo|parking|taxi"),
    ("Utilities", r"electricity|water bill|gas bill|broadband|wifi|recharge|mobile bill|dth"),
    ("Subscriptions", r"netflix|spotify|prime video|amazon prime|hotstar|youtube premium|subscription|icloud|gym membership"),
    ("Shopping", r"amazon|flipkart|myntra|mall|store purchase|shopping"),
    ("Entertainment", r"movie|cinema|pvr|inox|bookmyshow|concert|game"),
    ("Healthcare", r"pharmacy|hospital|clinic|doctor|medical|insurance premium|apollo"),
]
_COMPILED_RULES = [(cat, re.compile(pat, re.IGNORECASE)) for cat, pat in _RULES]


def classify_rule_based(description: str, amount: float) -> str:
    """Offline zero-shot-style classifier using keyword pattern matching."""
    text = description or ""
    for category, pattern in _COMPILED_RULES:
        if pattern.search(text):
            return category
    if amount is not None and amount > 0:
        return "Income"
    return "Miscellaneous"


def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Engine 2: Local zero-shot classifier (free, offline after first download)
# ---------------------------------------------------------------------------
# Uses a pretrained NLI model (facebook/bart-large-mnli) via HuggingFace
# `transformers`, run entirely on your own machine -- no API key, no cost,
# no internet needed after the one-time model download (~1.5GB).
# This is the textbook definition of "zero-shot classification": the model
# was never trained on our specific categories, it reasons about them at
# inference time from their plain-English labels alone.

_local_pipeline = None  # lazy-loaded singleton so the model loads only once


def _get_local_pipeline():
    global _local_pipeline
    if _local_pipeline is None:
        from transformers import pipeline
        _local_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _local_pipeline


def classify_zero_shot_local(description: str, amount: float) -> str:
    """Free, offline zero-shot classification using a local HuggingFace model.
    Falls back to the rule-based engine if `transformers`/`torch` aren't
    installed or the model fails to load for any reason."""
    try:
        clf = _get_local_pipeline()
        result = clf(description, candidate_labels=CATEGORIES)
        return result["labels"][0]
    except Exception:
        return classify_rule_based(description, amount)


def classify_with_llm(description: str, amount: float, client=None) -> str:
    """
    Zero-shot classification via the Anthropic API. Falls back to the
    rule-based engine automatically if no API key is configured, the
    `anthropic` package isn't installed, or the call fails for any reason
    (e.g. no network) -- this keeps the pipeline resilient for offline
    grading/demo environments.
    """
    if not _llm_available():
        return classify_rule_based(description, amount)

    try:
        import anthropic
        client = client or anthropic.Anthropic()
        prompt = (
            "Classify this bank transaction into exactly one category from "
            f"this list: {CATEGORIES}. Transaction description: \"{description}\", "
            f"amount: {amount}. Respond with only the category name, nothing else."
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        label = resp.content[0].text.strip()
        return label if label in CATEGORIES else classify_rule_based(description, amount)
    except Exception:
        return classify_rule_based(description, amount)


def classify_dataframe(df, engine: str = "rule"):
    """Add a `category` column and a `category_confidence` column to df.

    engine: "rule" (default, instant, free) | "local_zero_shot" (free,
    offline HuggingFace model, slower) | "llm" (Anthropic API, needs
    ANTHROPIC_API_KEY, fastest quality-per-call but costs money).

    confidence is 'high' when a rule/model matched cleanly, 'low' when the
    transaction fell through to Miscellaneous -- the Reflection step in
    the agent loop uses this to flag items for user review.
    """
    df = df.copy()
    engine_fn = {
        "rule": classify_rule_based,
        "local_zero_shot": classify_zero_shot_local,
        "llm": classify_with_llm,
    }.get(engine, classify_rule_based)

    categories = []
    confidences = []
    for _, row in df.iterrows():
        cat = engine_fn(row["description"], row["amount"])
        categories.append(cat)
        confidences.append("low" if cat == "Miscellaneous" else "high")
    df["category"] = categories
    df["category_confidence"] = confidences
    return df
