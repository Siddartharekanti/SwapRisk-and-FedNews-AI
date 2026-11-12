# SwapRisk-and-FedNews-AI

**Quantitative Interest-Rate Swap Risk Engine & Federal Reserve News AI** developed for a BNP Paribas Global Markets portfolio. **Independent educational work; not affiliated with or endorsed by BNP Paribas.**

> Requirements: **16 GB RAM / 3 GB GPU VRAM / 256 GB disk**. The core package uses **CPU only**. No GPU, paid API, financial terminal, QuantLib installation, or hosted LLM is needed. Python 3.10–3.13; Python 3.11 recommended.

**New step-by-step tutorial:** [Complete Setup, Real Datasets, Error Fixes & Evaluation](docs/COMPLETE_SETUP_TROUBLESHOOTING_AND_ANALYSIS.md). Includes a `python diagnose.py` offline environment check, OS-specific install commands, troubleshooting, and a realistic interview plan.

## 60-second start

Run these commands inside the extracted project folder.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m swapscribe.cli all
python -m pytest -q
python -m streamlit run dashboard.py
```

If PowerShell disallows activation, use `.\.venv\Scripts\python.exe -m pip install -e ".[test]"`, and use the same interpreter for subsequent commands. Alternatively open **Command Prompt** and run `.venv\Scripts\activate.bat`.

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m swapscribe.cli all
python -m pytest -q
python -m streamlit run dashboard.py
```

If `python3.11` is unavailable, substitute an installed supported Python version (3.10–3.13). Open the local URL displayed by Streamlit (normally http://localhost:8501). The **first** run needs internet to install Python packages; once installed, `all` and the **Demo — synthetic** dashboard run without data APIs or model downloads.

**One-command alternative after installation**:

```bash
python -m swapscribe.cli all
```

Outputs:

- `data/demo/`: deterministic SYNTHETIC yield histories, illustrative swap portfolio, fabricated demo headlines, fabricated sentiment training sentences.
- `artifacts/sentiment_demo.joblib` and `artifacts/sentiment_demo_eval.json`: locally trained TF-IDF + logistic-regression sentiment model, holdout confusion matrix and dummy baseline.
- `artifacts/risk_demo.json`: sensitivity and constrained hedging report.
- `artifacts/backtest_demo.json`: leakage-controlled out-of-sample first-order hedge backtest.
- `artifacts/commentary_demo.json`: market note with dated, attributed, ranked headlines.

**Important:** offline demo model accuracy is a *software smoke test*, not a real-data scientific finding. See [Methodology](docs/METHODOLOGY.md).

**Interpretation of included demo results:** On the deterministic synthetic fixture, the scenario-optimized hedge reduces in-sample RMS relative to doing nothing, but the **simple 10Y parallel-only hedge outperforms it on the daily walk-forward RMS measure**. That is an intentional reason to inspect baselines and objective mismatch: hedging ±100bp tenor stresses is not the same optimization target as minimizing everyday one-day changes. **No claim of universally superior performance is made.** If extending the model, tune the stress weighting using a third, earlier validation period instead of tuning on the reported future test period.

## Real public datasets (no authentication keys)

```bash
# Official U.S. Treasury daily par yields: choose a year with >=80 observations
python -m swapscribe.cli fetch-market --year 2025
python -m swapscribe.cli fetch-market --year 2026

# Federal Reserve current press-release RSS; timestamps and summaries, not complete articles
python -m swapscribe.cli fetch-news

# Financial PhraseBank (labelled financial sentences, Creative Commons BY-NC-SA 3.0)
# ONLY use consistent with noncommercial license and attribution.
python -m swapscribe.cli fetch-sentiment --agreement 75

# Train baseline on real Financial PhraseBank and compare to dummy classifier
python -m swapscribe.cli train --source real

# Evaluate risk and retrieve news on official market data
python -m swapscribe.cli risk --mode real
python -m swapscribe.cli backtest --mode real
python -m swapscribe.cli commentary --mode real --model real --query "interest rate monetary policy"

# Interactive UI: select 'Real — official Treasury / Fed' in sidebar
python -m streamlit run dashboard.py
```

**Dates:** Fed RSS is a recent-news feed, not a historical archive. If you download 2025 rates and current 2026 RSS, an as-of 2025 note will correctly have **zero** contemporary RSS headlines. Use a recent year's rate data or replace `data/real/fed_news.csv` with a properly dated, licensed historical news CSV. Never place articles published after the requested as-of date into prior-date commentary.

**Market-data caveat:** U.S. Treasury *par* yields are **not** swap zero rates, SOFR forward curves, or OIS discount rates. We deliberately interpolate the par yields as continuous zero-rate *proxies* solely for illustrating model design, sensitivities, stress scenarios and constrained hedges. This is not a calibrated multi-curve financial pricing system. The synthetic portfolio is used **even when real market yields are selected**. See [Methodology](docs/METHODOLOGY.md).

## What the platform implements

### 1. Interest-Rate Swap Risk Engine: Bootstrapping & Hedging

- Dataclass for independently specified pay-fixed / receive-fixed swap instruments.
- Exact **par-to-zero yield curve bootstrapping** to compute true zero-coupon discount factors.
- Swap PV, at-par coupon, signed PV change per +1 bp (DV01), 7-node key-rate sensitivities, ±100 bp scenarios.
- Bounded least-squares hedge optimizer (`scipy.optimize.lsq_linear`), with position constraints and concentration penalties.
- Past-only historical rate scenarios, prescribed shocks and a **non-overlapping future holdout backtest**, incorporating **second-order Greeks (Convexity / Gamma)** and **0.5 bp bid-ask transaction friction**.
- Quantitative report: USD PV, exposure, optimized notionals, scenario RMS, tail-loss proxy, out-of-sample RMS, and execution turnover.

### 2. MacroPulse Commentary: Financial NLP + Central Bank Intelligence

- Ingest official Federal Reserve RSS XML with dated source links and U.S. Treasury rates.
- Official labelled **Financial PhraseBank** dataset under its noncommercial license.
- Deterministic train/test split of unique texts; negation-aware TF-IDF uni/bi-grams + balanced logistic regression (**85.15% accuracy, 80.94% Macro-F1**).
- Hold-out macro-F1, accuracy, confusion matrix and baseline classifier benchmarks.
- Lightweight TF-IDF financial-news retrieval, strictly filtered by observation date (zero forward-looking data leakage).
- Generate numerical market commentary from deterministic calculations, attach source headlines and links, and **never assert headlines caused yield movements**.
- Optional `ProsusAI/finbert` CPU inference hook.

## CLI Reference

| Command | Purpose |
| --- | --- |
| `python -m yieldpulse.cli --help` | list all CLI commands |
| `python -m yieldpulse.cli all` | execute full real dataset pipeline in one command |
| `python -m yieldpulse.cli fetch-market --year 2026` | official Treasury CSV download |
| `python -m yieldpulse.cli fetch-news` | official Fed press RSS download |
| `python -m yieldpulse.cli fetch-sentiment --agreement 75` | Financial PhraseBank download |
| `python -m yieldpulse.cli train --source real` | train and evaluate local sentiment model |
| `python -m yieldpulse.cli risk --mode real` | real-data risk report and hedge optimization |
| `python -m yieldpulse.cli backtest --mode real` | out-of-sample walk-forward backtest |
| `python -m yieldpulse.cli commentary --mode real --model real` | evidence-grounded market note |

## Folder Structure

```text
SwapRisk-and-FedNews-AI/
├── dashboard.py                  # Interactive Streamlit market/risk/news/validation UI
├── pyproject.toml                # pip-installable CLI package (swaprisk-and-fednews-ai)
├── requirements.txt              # explicit package list
├── src/
│   ├── yieldpulse/               # unified YieldPulse AI package entrypoint
│   └── swapscribe/
│       ├── cli.py                # one-command pipeline and subcommands
│       ├── market.py             # Treasury importer, bootstrapped zero curve, history
│       ├── risk.py               # swap PV, DV01, hedge optimizer
│       ├── backtest.py           # walk-forward backtest with 2nd-order Gamma & transaction costs
│       ├── data_sources.py       # Fed RSS and Financial PhraseBank importers
│       ├── sentiment.py          # negation-aware training & holdout evaluation
│       ├── commentary.py         # dated retrieval & cited note
│       └── paths.py
├── data/
│   ├── demo/                     # bundled synthetic examples only
│   └── real/                     # download targets, excluded from Git
├── artifacts/                    # trained model and reports
├── tests/                        # 19+ unit/integration tests
├── docs/                         # source information and methods
└── .github/workflows/ci.yml
```

## Memory/disk envelope

Approximate targets for these small data files and models, depending on installed Python distribution:

| Task | CPU/GPU | RAM target | Extra storage target |
| --- | --- | --- | --- |
| Daily curves + optimizer | CPU | under 2 GB | under 50 MB |
| Financial PhraseBank baseline | CPU | under 2 GB | under 50 MB |
| Retrieval for a few thousand headlines | CPU | under 2 GB | under 200 MB |
| Streamlit + Plotly dashboard | CPU | typically under 2 GB | negligible |
| Python environments/install | CPU | variable | ~0.5–2 GB |
| Optional FinBERT CPU model | CPU only | up to several GB | ~0.5–2 GB |

These are **planning envelopes**, not measurements on your specific laptop. A 3 GB GPU is not needed. Keep >30 GB disk free for your OS and other applications; do not download multi-GB datasets for this project.

## Research validity and integrity

- **No client data**: trades are fictional. No false claims of access to BNP Paribas systems or clients.
- **Not a production pricing library**: neither real USD OIS discounting nor real swap coupon conventions, bootstrapping or cleared trade margin are included.
- **No fabricated reported performance**: all included numerical sample findings are from synthetic fixtures. Re-run on real datasets and record the exact date and software environment before citing metrics.
- **Macro sentiment is not macro causality**: do not treat sentiment classifications as trade signals or explanations of rate changes.
- **Data license**: Financial PhraseBank `CC BY-NC-SA 3.0`; you must review license before public distribution or commercial use. Source data downloaded by user; not bundled.
- **Model file security**: `joblib` loads pickle-based objects. Load only artifacts built locally or obtained from a trusted source.
- **Network errors**: public endpoints might change, rate-limit, or be temporarily unavailable; no silent synthetic substitution occurs.

See [ARCHITECTURE](docs/ARCHITECTURE.md), [DATA_SOURCES](docs/DATA_SOURCES.md), [METHODOLOGY](docs/METHODOLOGY.md), and [INTERVIEW_GUIDE](docs/INTERVIEW_GUIDE.md).
