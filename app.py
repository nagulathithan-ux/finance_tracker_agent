"""
Finance Tracker Agent - Interactive Dashboard Module
=======================================================
Entry point. Run with:  streamlit run app.py

Ties together the agentic loop:
  Perceive (ingest) -> Reason/Act (classify, budget check, anomaly scan)
  -> Reflect (flag low-confidence items) -> Output (dashboard + summary)
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.dirname(__file__))
from modules import data_ingestion, classifier, budget_engine, advisor, chat_advisor

st.set_page_config(page_title="Finance Tracker Agent", layout="wide")
st.title("💰 Finance Tracker Agent")
st.caption("An agentic pipeline: ingest → classify → budget-check → detect anomalies → advise")

# ---------------- Sidebar: input + settings ----------------
with st.sidebar:
    st.header("1. Data source")
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled sample dataset", value=uploaded is None)

    st.header("2. Classification engine")
    engine_choice = st.radio(
        "How should transactions be categorized?",
        options=["Rule-based (instant, free)",
                 "Local zero-shot AI (free, offline, ~1-2s/transaction)",
                 "LLM API (Anthropic, needs ANTHROPIC_API_KEY, paid)"],
        index=0,
    )
    engine = {
        "Rule-based (instant, free)": "rule",
        "Local zero-shot AI (free, offline, ~1-2s/transaction)": "local_zero_shot",
        "LLM API (Anthropic, needs ANTHROPIC_API_KEY, paid)": "llm",
    }[engine_choice]

    st.header("3. Monthly budgets (₹)")
    budgets = {}
    for cat, default in budget_engine.DEFAULT_BUDGETS.items():
        budgets[cat] = st.number_input(cat, min_value=0, value=default, step=500)

    st.header("4. Chat advisor engine")
    if chat_advisor.groq_available():
        chat_engine = st.radio(
            "How should chat questions be answered?",
            options=["Rule-based (instant, free)",
                     "Groq LLM (free tier, needs internet)"],
            index=1,
        )
    else:
        chat_engine = "Rule-based (instant, free)"
        st.caption("Set GROQ_API_KEY as an environment variable to unlock the "
                   "free Groq-powered chat advisor. Using rule-based advisor for now.")

# ---------------- Load data ----------------
source_path = None
if uploaded is not None:
    source_path = uploaded
elif use_sample:
    source_path = os.path.join(os.path.dirname(__file__), "data", "sample_transactions.csv")

if source_path is None:
    st.info("Upload a CSV or check 'Use bundled sample dataset' to get started.")
    st.stop()

try:
    raw_df = pd.read_csv(source_path)
    raw_df.columns = [c.strip().lower() for c in raw_df.columns]
    df = data_ingestion.clean_transactions(raw_df)
except Exception as e:
    st.error(f"Could not process the file: {e}")
    st.stop()

report = df.attrs.get("cleaning_report", {})
with st.expander("🧹 Data Ingestion report", expanded=False):
    st.json(report)

# ---------------- Classify ----------------
if engine == "local_zero_shot":
    with st.spinner(f"Running local zero-shot model on {len(df)} transactions "
                     f"(first run also downloads the model, ~1.5GB — this can take a while)..."):
        df = classifier.classify_dataframe(df, engine=engine)
else:
    df = classifier.classify_dataframe(df, engine=engine)

# ---------------- Budget & anomaly engine ----------------
alerts = budget_engine.check_budget_alerts(df, budgets)
anomalies = budget_engine.detect_anomalies(df)
recurring = budget_engine.detect_recurring_subscriptions(df)

# ---------------- Metric cards ----------------
income = df[df["amount"] > 0]["amount"].sum()
expenses = df[df["amount"] < 0]["amount"].abs().sum()
net = income - expenses

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Income", f"₹{income:,.0f}")
c2.metric("Total Expenses", f"₹{expenses:,.0f}")
c3.metric("Net Cash Flow", f"₹{net:,.0f}")
c4.metric("Transactions", len(df))

# ---------------- Charts ----------------
col1, col2 = st.columns(2)
with col1:
    cat_spend = (
        df[df["amount"] < 0].groupby("category")["amount"]
        .apply(lambda s: s.abs().sum()).reset_index(name="spend")
        .sort_values("spend", ascending=False)
    )
    fig = px.pie(cat_spend, names="category", values="spend", title="Spending by Category", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    monthly = budget_engine.monthly_category_spend(df)
    fig2 = px.bar(monthly, x="month", y="spend", color="category", title="Monthly Spend by Category")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Budget alerts ----------------
st.subheader("⚠ Budget Alerts")
if alerts:
    st.dataframe(pd.DataFrame(alerts), use_container_width=True)
else:
    st.success("All categories are within budget.")

# ---------------- Anomalies ----------------
st.subheader("🚨 Anomalous / Spike Transactions")
if len(anomalies):
    st.dataframe(anomalies, use_container_width=True)
else:
    st.success("No abnormal transactions detected.")

# ---------------- Recurring subscriptions ----------------
st.subheader("↻ Detected Recurring Subscriptions")
if len(recurring):
    st.dataframe(recurring, use_container_width=True)
else:
    st.info("No recurring charges detected yet.")

# ---------------- Low-confidence categorizations (reflection step) ----------------
low_conf = df[df["category_confidence"] == "low"]
if len(low_conf):
    st.subheader("🔎 Needs Review (low-confidence categorization)")
    st.dataframe(low_conf[["date", "description", "amount", "category"]], use_container_width=True)

# ---------------- Advisor summary ----------------
st.subheader("🧠 AI Summary & Recommendations")
st.text(advisor.generate_summary(df, alerts, anomalies, recurring))

# ---------------- Financial Health: where/when to spend or save ----------------
st.subheader("🩺 Financial Health: Where & When to Spend or Save")
health = advisor.financial_health_report(df, alerts, anomalies, recurring, budgets)
hc1, hc2 = st.columns(2)
with hc1:
    st.markdown("**📉 Where to cut back**")
    for line in health["where_to_cut"]:
        st.markdown(f"- {line}")
    st.markdown("**💡 How to save more**")
    for line in health["how_to_save_more"]:
        st.markdown(f"- {line}")
with hc2:
    st.markdown("**✅ Where it's safe to keep spending**")
    for line in health["where_safe_to_spend"]:
        st.markdown(f"- {line}")
    st.markdown("**⏱ Is now a good time to spend?**")
    for line in health["when_to_spend"]:
        st.markdown(f"- {line}")

# ---------------- Conversational chat advisor ----------------
st.subheader("💬 Ask your Finance Agent")
st.caption("e.g. 'Where can I cut back this month?', 'Is now a good time to buy a laptop?', "
           "'How much did I spend on Food & Dining?'")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def _rule_based_answer(question: str) -> str:
    ql = question.lower()
    matched = [c for c in classifier.CATEGORIES if c.lower() in ql]
    if matched:
        total = df[(df["category"] == matched[0]) & (df["amount"] < 0)]["amount"].abs().sum()
        return f"You spent ₹{total:,.0f} on {matched[0]} in this dataset."
    if "income" in ql or "earn" in ql:
        return f"Total income in this dataset: ₹{income:,.0f}"
    if "save" in ql or "net" in ql:
        return f"Net cash flow (income − expenses): ₹{net:,.0f}"
    if "cut" in ql or "reduce" in ql or "trim" in ql:
        return " ".join(health["where_to_cut"])
    if "good time" in ql or "should i buy" in ql or "afford" in ql:
        return " ".join(health["when_to_spend"])
    if "save more" in ql or "how to save" in ql:
        return " ".join(health["how_to_save_more"])
    return ("I can answer questions about a specific category, income, net savings, "
            "where to cut back, or whether it's a good time to spend. "
            "Try: 'Where can I cut back?'")

question = st.chat_input("Ask a question about your finances...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    use_groq = chat_engine == "Groq LLM (free tier, needs internet)"
    answer = None
    if use_groq:
        try:
            with st.spinner("Thinking..."):
                answer = chat_advisor.ask(
                    question, df, alerts, anomalies, recurring, budgets,
                    history=st.session_state.chat_history[:-1],
                )
        except Exception as e:
            answer = None
            st.caption(f"(Groq call failed, falling back to rule-based: {e})")

    if answer is None:
        answer = _rule_based_answer(question)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

with st.expander("View full cleaned & categorized transaction table"):
    st.dataframe(df, use_container_width=True)
