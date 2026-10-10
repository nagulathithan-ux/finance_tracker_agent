"""
Budget & Anomaly Engine
-------------------------
Compares category-wise spending against user-defined monthly budgets and
raises threshold alerts, and flags statistically abnormal transactions
(spikes) and likely recurring subscriptions.
"""

import pandas as pd
import numpy as np

DEFAULT_BUDGETS = {
    "Groceries": 8000, "Food & Dining": 5000, "Transportation": 4000,
    "Utilities": 3500, "Subscriptions": 1500, "Shopping": 6000,
    "Entertainment": 3000, "Healthcare": 4000, "Rent/Housing": 15000,
    "Miscellaneous": 2000,
}


def monthly_category_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Total spend (debits only) per category per month."""
    expenses = df[df["amount"] < 0].copy()
    expenses["month"] = expenses["date"].dt.to_period("M").astype(str)
    expenses["spend"] = expenses["amount"].abs()
    return expenses.groupby(["month", "category"])["spend"].sum().reset_index()


def check_budget_alerts(df: pd.DataFrame, budgets: dict = None) -> list:
    """Return a list of alert dicts for categories that exceed their budget
    in any month present in the data."""
    budgets = budgets or DEFAULT_BUDGETS
    spend = monthly_category_spend(df)
    alerts = []
    for _, row in spend.iterrows():
        limit = budgets.get(row["category"])
        if limit and row["spend"] > limit:
            pct = round((row["spend"] / limit - 1) * 100, 1)
            alerts.append({
                "month": row["month"],
                "category": row["category"],
                "spent": round(row["spend"], 2),
                "budget": limit,
                "over_by_pct": pct,
                "message": f"{row['category']} spending in {row['month']} is "
                           f"₹{row['spend']:.0f}, {pct}% over the ₹{limit} budget.",
            })
    return sorted(alerts, key=lambda a: -a["over_by_pct"])


def detect_anomalies(df: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """Flag transactions whose amount is a statistical outlier within its
    own category (z-score based spike detection)."""
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()

    grp = expenses.groupby("category")["abs_amount"]
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(0)
    count = grp.transform("count")

    z = np.where((count >= 3) & (std > 0), (expenses["abs_amount"] - mean) / std, 0.0)
    expenses["z_score"] = z

    anomalies = expenses[expenses["z_score"] >= z_thresh].copy()
    return anomalies.sort_values("z_score", ascending=False)[
        ["date", "description", "category", "abs_amount", "z_score"]
    ]


def detect_recurring_subscriptions(df: pd.DataFrame, min_occurrences: int = 2) -> pd.DataFrame:
    """Heuristic: same description + same (rounded) amount recurring on
    a roughly monthly cadence => likely subscription/recurring charge."""
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs().round(0)
    grouped = expenses.groupby(["description", "abs_amount"]).agg(
        occurrences=("date", "count"),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
    ).reset_index()
    recurring = grouped[grouped["occurrences"] >= min_occurrences]
    return recurring.sort_values("occurrences", ascending=False)
