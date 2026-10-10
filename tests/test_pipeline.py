"""
End-to-end test of the Finance Tracker Agent pipeline.
Run: python3 tests/test_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules import data_ingestion, classifier, budget_engine, advisor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.csv")


def main():
    print("=" * 70)
    print("MODULE 1: Data Ingestion")
    print("=" * 70)
    df = data_ingestion.load_transactions(DATA_PATH)
    print("Cleaning report:", df.attrs["cleaning_report"])
    assert len(df) > 0, "No rows survived ingestion!"
    assert df["date"].isna().sum() == 0, "Unparsed dates leaked through!"
    print(f"PASS: {len(df)} clean rows loaded.\n")

    print("=" * 70)
    print("MODULE 2: AI Classification (rule-based engine)")
    print("=" * 70)
    df = classifier.classify_dataframe(df, engine="rule")
    assert "category" in df.columns
    counts = df["category"].value_counts()
    print(counts.to_string())
    low_conf = (df["category_confidence"] == "low").sum()
    print(f"Low-confidence (Miscellaneous) categorizations: {low_conf}/{len(df)}")
    assert low_conf / len(df) < 0.2, "Too many uncategorized transactions -- rule set needs work."
    print("PASS: categorization within acceptable Miscellaneous rate.\n")

    print("=" * 70)
    print("MODULE 3: Budget & Anomaly Engine")
    print("=" * 70)
    alerts = budget_engine.check_budget_alerts(df)
    print(f"Budget alerts raised: {len(alerts)}")
    for a in alerts:
        print(" -", a["message"])

    anomalies = budget_engine.detect_anomalies(df)
    print(f"\nAnomalous transactions flagged: {len(anomalies)}")
    print(anomalies.to_string(index=False))
    assert "LAPTOP" in anomalies["description"].str.cat(sep=" "), \
        "Expected the planted ₹68,000 laptop purchase to be flagged as an anomaly!"
    print("PASS: known planted anomaly was correctly detected.")

    recurring = budget_engine.detect_recurring_subscriptions(df)
    print(f"\nRecurring subscriptions detected: {len(recurring)}")
    print(recurring.to_string(index=False))
    assert len(recurring) >= 3, "Expected Netflix/Spotify/Prime to be detected as recurring!"
    print("PASS: known recurring subscriptions correctly detected.\n")

    print("=" * 70)
    print("MODULE 4: Advisor summary (feeds the Dashboard Module)")
    print("=" * 70)
    summary = advisor.generate_summary(df, alerts, anomalies, recurring)
    print(summary)
    assert "Total income" in summary
    print("\nPASS: summary generated successfully.\n")

    print("=" * 70)
    print("MODULE 4b: Financial Health report (rule-based save/spend/timing advice)")
    print("=" * 70)
    health = advisor.financial_health_report(df, alerts, anomalies, recurring,
                                              budget_engine.DEFAULT_BUDGETS)
    for key in ["where_to_cut", "where_safe_to_spend", "when_to_spend", "how_to_save_more"]:
        print(f"{key}:")
        for line in health[key]:
            print("  -", line)
    assert all(k in health for k in ["where_to_cut", "where_safe_to_spend",
                                      "when_to_spend", "how_to_save_more"])
    print("\nPASS: financial health report generated successfully.\n")

    print("=" * 70)
    print("ALL MODULES EXECUTED AND TESTED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
