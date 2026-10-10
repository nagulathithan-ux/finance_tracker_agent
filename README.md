# Finance Tracker Agent

An agentic personal-finance pipeline: **ingest → classify → budget-check /
anomaly-detect → advise → dashboard**, matching the 4-module architecture
in the project PPT (Data Ingestion, AI Classification, Budget & Anomaly
Engine, Interactive Dashboard).

## Project structure

```
finance_tracker_agent/
├── app.py                     # Streamlit dashboard (main entry point)
├── modules/
│   ├── data_ingestion.py      # Module 1: load & clean raw CSVs
│   ├── classifier.py          # Module 2: LLM / rule-based categorization
│   ├── budget_engine.py       # Module 3: budget alerts + anomaly detection
│   └── advisor.py             # Summary & recommendation generator
├── data/
│   ├── sample_transactions.csv
│   └── make_sample_data.py    # regenerates the sample dataset
├── tests/
│   └── test_pipeline.py       # end-to-end test of all 4 modules
├── requirements.txt
└── README.md
```

## Classification engines

The app now offers three interchangeable engines (pick one in the sidebar):

| Engine | Cost | Speed | Needs internet? | How it works |
|---|---|---|---|---|
| **Rule-based** (default) | Free | Instant | No | Keyword pattern matching |
| **Local zero-shot AI** | Free | ~1-2s/transaction | Only for first-time model download (~1.5GB) | Runs `facebook/bart-large-mnli` on your own CPU via HuggingFace `transformers` — genuine zero-shot classification, entirely offline after setup |
| **LLM API** | Paid (fractions of a cent/call) | Fast | Yes, every run | Calls Claude via the Anthropic API |

All three share the same fallback safety net: if a model/API isn't
available for any reason, classification silently drops back to the
rule-based engine rather than crashing the app.

## Chat advisor engine

The "Ask your Finance Agent" chat box (and the Financial Health panel above
it) works two ways:

| Engine | Cost | How it works |
|---|---|---|
| **Rule-based** (default, always available) | Free | Template logic reasoning over your actual budgets, spending, anomalies, and recurring charges |
| **Groq LLM** (optional) | Free tier, no credit card | Sends your real spending summary to Groq's `llama-3.3-70b-versatile` model for genuinely open-ended, conversational answers |

To enable the Groq chat engine:

1. Sign up free at [console.groq.com](https://console.groq.com) (no credit card needed) and create an API key
2. Set it as an environment variable before launching:
   ```bash
   export GROQ_API_KEY=your_key_here      # Mac/Linux
   $env:GROQ_API_KEY="your_key_here"      # Windows PowerShell
   ```
3. Run `streamlit run app.py` — a "Chat advisor engine" option appears in the sidebar

If the key isn't set, or the API call fails for any reason (no internet,
rate limit, etc.), the chat automatically falls back to the rule-based
engine mid-conversation — it never crashes.

**A note on caution:** avoid third-party "free shared API key" sites you
might find online (key-aggregator repos, unofficial proxy services). They
route your requests through an unknown third party, violate the original
provider's terms, and the keys get revoked constantly. Groq's own official
free tier (above) is the only source used here.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens with the bundled sample dataset pre-loaded so you can demo
it immediately, or upload your own `date,description,amount` CSV in the
sidebar.

To enable real LLM zero-shot classification instead of the rule-based
engine, set an API key before launching:

```bash
export ANTHROPIC_API_KEY=your_key_here
streamlit run app.py   # then toggle "Use LLM zero-shot classifier" on
```

If the key isn't set (or the API call fails for any reason, e.g. no
network), the app automatically falls back to the rule-based engine —
the pipeline never breaks either way.

## Running the test suite

```bash
python3 tests/test_pipeline.py
```

## Test results (last run, on the bundled sample data)

All four modules were executed end-to-end against 62 synthetic
transactions (deliberately including a bad date, a missing amount, a
duplicate row, a ₹68,000 anomaly purchase, and monthly Netflix/Spotify/
Prime/rent/utility charges), and every module passed:

| Module | What was tested | Result |
|---|---|---|
| Data Ingestion | Bad date row, missing-amount row, and duplicate row | All 3 correctly dropped/cleaned; 59/62 clean rows kept |
| AI Classification (rule engine) | 59 transactions across 10 categories | 0% fell through to "Miscellaneous" |
| Budget & Anomaly Engine | Monthly category limits vs. actual spend | 4 real budget breaches correctly flagged (e.g. Shopping 1128.8% over) |
| Budget & Anomaly Engine | Planted ₹68,000 laptop purchase | Correctly flagged via z-score (z ≈ 2.27) |
| Budget & Anomaly Engine | Netflix / Spotify / Prime / Rent / Broadband | All 5 correctly identified as recurring |
| Advisor | Natural-language summary generation | Correctly reported net cash flow of −₹9,168 and recommended trimming Shopping |

Re-run `python3 tests/test_pipeline.py` any time to reproduce this.

**Note on the Streamlit UI itself:** `app.py` was verified for correctness
via `python3 -m py_compile` (clean) and by exercising the exact same
module calls it uses, in `test_pipeline.py`. The dashboard's browser UI
could not be launched inside this sandboxed environment because it has
no network access, so `pip install streamlit plotly` was not possible
here — install them in your own machine (`pip install -r
requirements.txt`) and run `streamlit run app.py` there; it uses the
same, already-tested backend logic.

**Note on the Local zero-shot AI engine specifically:** it could not be
downloaded or test-run in this sandbox for the same no-internet reason.
The code path has the same try/except fallback pattern as the LLM engine
(if `transformers`/`torch` aren't installed, or the model fails to load,
it silently falls back to the rule-based engine), so the app stays safe
to demo either way — but test it once on your own machine *before* your
guide meeting so the first (slow) model download happens ahead of time,
not live during the demo.

## Limitations (be upfront about these to your guide)

- **No live bank integration** — this is a batch CSV pipeline, not a
  real-time Open Banking / bank-API connector (matches the "Future
  Scope" item on your Conclusion slide).
- **Rule-based classifier accuracy** — the offline fallback uses keyword
  matching, so unfamiliar merchant names (not in the rule list) fall
  into "Miscellaneous" and need the LLM engine or a larger rule set for
  full zero-shot coverage.
- **LLM engine needs an API key + internet** — without one, the app
  silently uses the rule-based engine; there's no visual warning banner
  for this yet.
- **Anomaly detection is statistical, not learned** — z-score outlier
  detection per category needs at least 3 transactions in a category to
  be meaningful; categories with very few transactions won't get
  reliable anomaly flags.
- **Recurring-subscription detection is heuristic** — it matches on
  identical description + rounded amount, so a subscription that
  changes price (e.g. a promo ending) won't be linked across months.
- **Single-user, single-currency, local-only** — no multi-user auth, no
  database (data lives only for the browser session), no multi-currency
  support.
- **Financial health advice is heuristic, not personalized financial
  planning** — the "where to cut" / "when to spend" logic uses simple
  rules (budget overages, discretionary-category share, savings rate
  thresholds) grounded in your real numbers, but it isn't a substitute
  for professional financial advice, and the "where to cut" list compares
  total spend across the whole dataset against a single month's budget,
  so it can look more alarming than it is on multi-month sample data.
- **The Groq chat advisor needs internet and a free API key** — without
  one, it automatically uses the same rule-based engine, just with less
  flexible, template-based answers.

## Extending it

- Swap `advisor.generate_summary` for an LLM call (same fallback pattern
  as `classifier.py`) for richer, more varied recommendations.
- Add a persistence layer (SQLite/Postgres) to track budgets and history
  across sessions instead of re-uploading a CSV each time.
- Expand `_RULES` in `classifier.py` or fine-tune a small local model to
  reduce reliance on the LLM API for classification.
