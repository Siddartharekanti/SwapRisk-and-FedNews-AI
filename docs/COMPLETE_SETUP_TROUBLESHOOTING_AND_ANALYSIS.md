# Full Runbook — SwapRisk-and-FedNews-AI

A beginner-to-interview guide for the **BNP Paribas Global Markets Data Science Intern** portfolio project. This independent educational project is **not endorsed by BNP Paribas**, uses a **synthetic swap portfolio**, and **does not use real bank client data**. The first runs can be done fully offline after pip installation.

Hardware assumptions: **16 GB RAM**, **3 GB VRAM** (unused by default), **256 GB disk** with at least **30 GB free**. Python **3.11** recommended; Python **3.10–3.13** supported. The baseline is scikit-learn and SciPy, not a 7B+ LLM and not QuantLib.

## 1. What you will deliver

- **SwapRisk** — ingest time series of U.S. Treasury par yields, create an illustrative synthetic fixed/floating swap book, calculate PV and per-tenor sensitivity, optimize 2Y/5Y/10Y offsetting hedge notionals, and compare against an unhedged book and a single 10Y DV01 hedge with walk-forward dates.
- **FedNews-AI** — ingest Financial PhraseBank and Federal Reserve RSS metadata, train a 3-class TF-IDF + balanced Logistic Regression sentiment model, evaluate against a majority-class baseline, retrieve as-of relevant headlines, and write a factual yield-curve market note with source links.
- **Dashboard** — view market curves, risk reports, optimizer notionals, out-of-sample results, market commentary, model metrics, and downloadable JSON.
- **Tests** — 19 baseline tests check pricing identities, training, cutoff handling, and integration. Tests do not prove financial production validity.

### Architecture

```text
                         [Data sources]
                   /                          \
        Treasury par yields                 Financial PhraseBank
                   |                        + Fed press RSS
              dated rate CSV                      |
                   |                     train / retrieve headlines
       +-----------+-----------+                   |
       |                       |          3-class sentiment model
  Curve proxy               Rate shocks             |
       |                       |            date-filtered retrieval
  Synthetic swap PV      Hedge optimization         |
       |                       |                Market note
       +-----------+-----------+-------------------+
                           |
                  Streamlit dashboard
```

## 2. Extract the ZIP correctly

Download `SwapShield_Q_FedNews_Lite_Complete.zip` (or this newer guided ZIP). Extract it to a path with no awkward special characters:

- Windows example: `C:\Projects\SwapRisk-and-FedNews-AI`
- macOS/Linux example: `~/Projects/SwapRisk-and-FedNews-AI`

The project root is the directory containing **`pyproject.toml`**, **`dashboard.py`**, **`diagnose.py`**, **`src/`**, and **`data/`**. Do not run the CLI from the directory *above* it.

## 3. Install Python 3.11 and packages

### Windows — easiest approach (Command Prompt)

Install Python 3.11 from https://www.python.org/downloads/ (ensure Python Launcher `py` is available), open **Command Prompt**, navigate to the root:

```bat
cd /d C:\Projects\SwapRisk-and-FedNews-AI
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[test]"
.venv\Scripts\python.exe diagnose.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m swapscribe.cli all
.venv\Scripts\python.exe -m streamlit run dashboard.py
```

All commands above reference the virtual-environment interpreter directly. No PowerShell activation is needed. If you use PowerShell, commands are equivalent but use `./.venv/Scripts/python.exe` or `.\.venv\Scripts\python.exe` with correct PowerShell quoting. For paths containing spaces, wrap the `cd` directory in double quotes.

### macOS/Linux

```bash
cd ~/Projects/SwapRisk-and-FedNews-AI
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python diagnose.py
python -m pytest -q
python -m swapscribe.cli all
python -m streamlit run dashboard.py
```

If `python3.11` is not installed but you have a supported 3.10, 3.12, or 3.13 version, substitute its command. On Apple Silicon, use the native ARM64 Python install when possible. GPU support is not needed. The dashboard ordinarily opens at `http://localhost:8501`; stop it with **Ctrl+C**.

**Successful baseline**: the diagnostic reports Python and dependencies as OK; pytest reports `19 passed`; the CLI prints `"status": "success"` and `"demo_is_synthetic": true`. The dashboard displays four tabs.

### Check folder and terminal quickly

```bash
python -m swapscribe.cli --help
python -m pip --version
python --version
```

If the CLI cannot import `swapscribe`, repeat `python -m pip install -e '.[test]'` in **the same virtual environment** that runs the project.

## 4. Dataset guide and exact CSV formats

**Dataset A — official U.S. Treasury par yield rates.** Source: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve. Use 2025 full-year data for a first backtest. The downloader converts to `data/real/treasury_2025.csv` with fields `date, 3 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, 30 Yr`. The `date` is `YYYY-MM-DD`, and yield values are **annual percentage points** (e.g., `4.25` means 4.25%, not `0.0425`). The algorithm converts them to decimal rates internally. Treasury *par* yields are not swap zero rates; the model is an **educational proxy only**.

**Dataset B — Financial PhraseBank.** Source: https://huggingface.co/datasets/takala/financial_phrasebank. Use `75` agreement (about 3,453 rows before local duplicate removal); three classes: `positive`, `negative`, `neutral`. File: `data/real/sentiment_train.csv`; columns: `text,label`. Noncommercial license **CC BY-NC-SA 3.0**; read and follow the license and attribution requirements before sharing data, weights or derived artifacts publicly. The dataset does not have official date-based splits.

**Dataset C — Fed press-release RSS.** Source: https://www.federalreserve.gov/feeds/feeds.htm. Downloaded as `data/real/fed_news.csv` with columns `date,title,summary,url,source,is_synthetic`. These are dated *headlines/summaries*, not complete licensed news articles. The RSS feed is recent, not a historical archive; publication timestamps on a date do not prove a story was available before the same day's market close.

**Dataset D — fictional swap portfolio.** Bundled `data/demo/portfolio.csv` contains `swap_id,notional_usd,maturity_years,fixed_rate_pct,direction`. Valid `direction`: `pay_fixed` or `receive_fixed`. `notional_usd` is in *USD* (e.g. `9000000`, NOT `9` for nine million), `fixed_rate_pct` uses annual percent numbers (e.g. `4.1` means 4.1%), and `maturity_years` must be a positive integer. It is still synthetic when real market yields are used.

**Demo inputs** already exist under `data/demo/`: `market_rates.csv`, `portfolio.csv`, `news.csv`, `sentiment_train.csv`. They are generated synthetic fixtures; do not cite their model performance as real-world research evidence.

## 5. First pipeline run: completely offline

From the project root, with venv Python activated (or prefix commands with the direct venv Python executable):

```bash
python -m swapscribe.cli demo
python -m swapscribe.cli train --source demo
python -m swapscribe.cli risk --mode demo
python -m swapscribe.cli backtest --mode demo
python -m swapscribe.cli commentary --mode demo --query "interest rates monetary policy"
python -m swapscribe.cli evaluate --mode demo
python -m pytest -q
python -m streamlit run dashboard.py
```

Or replace the first 6 steps with **`python -m swapscribe.cli all`**. Every `all` run regenerates `data/demo/` (so back up your edited demo portfolio first).

Outputs under `artifacts/`: `sentiment_demo.joblib`, `sentiment_demo_eval.json`, `risk_demo.json`, `backtest_demo.json`, `commentary_demo.json`. The CLI prints JSON directly to the terminal; the dashboard can export risk, walk-forward and cited-note JSON.

## 6. Real public data run (internet required for downloads)

Start with **one year**, 2025, for enough observations and simpler checks:

```bash
python -m swapscribe.cli fetch-market --year 2025
python -m swapscribe.cli backtest --mode real
python -m swapscribe.cli risk --mode real
```

Then download and train text sentiment separately:

```bash
python -m swapscribe.cli fetch-sentiment --agreement 75
python -m swapscribe.cli train --source real
```

For **current** Fed RSS commentary, also fetch the current year's Treasury rates. Current in this guide: November 2026. The CLI selects the **alphabetically latest** `treasury_*.csv` file when in real mode (so `treasury_2026.csv` wins over `treasury_2025.csv`):

```bash
python -m swapscribe.cli fetch-market --year 2026
python -m swapscribe.cli fetch-news
python -m swapscribe.cli commentary --mode real --model real --query "inflation interest rate monetary policy"
python -m swapscribe.cli backtest --mode real
python -m streamlit run dashboard.py
```

In the Streamlit sidebar choose **Dataset provenance → Real — official Treasury / Fed**. This selection appears only after at least one `data/real/treasury_*.csv` file exists. You will still see a **synthetic** swap book (there is no proprietary position download). If there are no news results, check that relevant news dates are **on or before** the selected market as-of date and that the RSS endpoint returned items.

**Important year-selection limitation:** There is not yet a CLI `--year` option for `risk`/`backtest` and no specific year dropdown in the dashboard. Real mode always picks the latest downloaded `treasury_YYYY.csv` filename. If you want to backtest only the 2025 file while retaining 2026 data, temporarily move `data/real/treasury_2026.csv` outside the `data/real` folder. Do not silently concatenate years without retesting data completeness and changes.

### Inspect local datasets

```bash
python -c "import pandas as pd; x=pd.read_csv('data/real/treasury_2025.csv'); print(x.shape); print(x.columns.tolist()); print(x.head(3).to_string(index=False))"
python -c "import pandas as pd; x=pd.read_csv('data/real/sentiment_train.csv'); print(x.shape); print(x.label.value_counts())"
python -c "import pandas as pd; x=pd.read_csv('data/real/fed_news.csv'); print(x.shape); print(x[['date','title']].head(5).to_string(index=False))"
```

### If automated downloads fail: manual official-source import

You may download a year CSV manually from the official Treasury site and put it into your project root as `treasury_raw.csv`. Use the **project's own normalizer** to produce the correct format:

```bash
python -c "from pathlib import Path; from swapscribe.market import clean_treasury_csv; import pandas as pd; p=Path('data/real/treasury_2025.csv'); p.parent.mkdir(exist_ok=True,parents=True); clean_treasury_csv(Path('treasury_raw.csv').read_text(encoding='utf-8-sig')).to_csv(p,index=False); print(p)"
```

If you download Financial PhraseBank manually, prefer the official ZIP and run its included parser rather than guessing columns:

```bash
python -c "from pathlib import Path; from swapscribe.data_sources import parse_phrasebank_zip; p=Path('data/real/sentiment_train.csv'); p.parent.mkdir(exist_ok=True,parents=True); parse_phrasebank_zip(Path('FinancialPhraseBank-v1.0.zip').read_bytes(),agreement='75').to_csv(p,index=False); print(p)"
```

For manually imported news, preserve its actual release dates and cite original URLs. Use exactly `date,title,summary,url,source` (and optionally `is_synthetic`). Do not invent news timestamps or use 2026 items in historical 2025 commentary.

## 7. How SwapRisk works (with equations)

1. Read yield history and choose a valuation date; keep only observations on or before that date.
2. Build a curve by interpolating Treasury par yields, treated *for demonstration only* as continuous zero-rate proxies, `z_proxy(t)`.
3. Approximate a discount factor as **`D(t) = exp(-z_proxy(t) * t)`**.
4. For a pay-fixed receive-floating swap with semiannual payments, approximate the fixed annuity as **`A(T) = sum[0.5 * D(t_i)]`** and the floating leg PV as **`N * (1 - D(T))`** at a reset date.
5. Approximate **`PV_payfixed = N * (1 - D(T)) - N * (k/100) * A(T)`**; receive-fixed reverses sign. These formulas ignore actual swap reset schedule, collateral/credit, stubs, coupons already accrued, and curve bootstrapping.
6. Compute signed parallel **DV01-style `PV(curve+0.5bp)-PV(curve-0.5bp)`** (USD change for +1bp). A basis point (bp) is 0.01 percentage point. `100 bp = 1 percentage point`.
7. Repeat node-by-node at **1Y,2Y,3Y,5Y,7Y,10Y,20Y** to obtain key-rate sensitivities (risk by maturity).
8. Construct past daily rate differences and **14** prescribed shocks: `+100bp` and `-100bp` separately for each of seven tenors.
9. Optimize signed 2Y,5Y,10Y reference hedge amounts (`x` in millions) via bounded regularized least squares: **minimize `||S(r + Hx)||²/n + lambda*||x||²`**, with `-30 <= x_i <= 30` million by default. Positive hedge notional = pay fixed, negative = receive fixed.
10. Report scenario RMS and worst scenario, and evaluate a **future** daily-return window using past-only hedge parameters, comparing: unhedged, simple 10Y parallel-DV01 hedge, and optimized multi-tenor hedge.

**How to interpret:** A lower scenario RMS after optimizing does *not* prove that the method beats a simpler hedge on future daily returns. The stress-weighted training objective and the live future-test metric differ; report both.

## 8. How FedNews-AI works (real NLP)

1. Read `sentiment_train.csv` with `text,label`.
2. Validate all 3 classes, trim strings, remove duplicate sentences, and fail on conflicting duplicate labels.
3. Stratify into 75% training and 25% testing (`random_state=42`). PhraseBank has no authoritative temporal split.
4. Build TF-IDF unigram/bigram vectors (up to 12,000 features), then fit a class-balanced logistic-regression classifier.
5. Compare held-out **accuracy**, **macro-F1**, and confusion matrix to a most-frequent dummy classifier. Macro-F1 gives equal importance to positive, neutral, and negative, unlike plain accuracy on an imbalanced dataset.
6. Read Fed RSS items. Filter out items dated *after* the requested cutoff date.
7. Use TF-IDF + cosine similarity to rank headline and description text for queries like `inflation interest rate monetary policy`.
8. Calculate 2Y and 10Y Treasury yields, changes in basis points, and the 10Y−2Y spread from actual market columns.
9. Build a deterministic cited note. Retrieved news items are **context**, not proof of what caused a rate change.

**What this package does not do:** It does not fine-tune FinBERT in the default pipeline; the optional `finbert_cpu` function is inference only. It does not run a generative LLM, a vector database, LangChain or a production multi-curve pricing stack. These may be sensible future enhancements, but don't claim them on a résumé until implemented and evaluated.

## 9. Dashboard operating instructions

Run `python -m streamlit run dashboard.py`.

- **Sidebar — Dataset provenance:** `Demo — synthetic` or `Real — official Treasury / Fed` (only when downloaded).
- **Sidebar — As-of market date:** choose a date with >=7 rate observations available. This prevents future observations from entering valuation or commentary.
- **Sidebar — Max absolute hedge per tenor ($MM):** default `30` means a hedge size limit of $30 million for *each* reference maturity.
- **Market & curves:** view 2Y, 10Y, 10Y–2Y spread, rate time series and curve shape.
- **SwapRisk risk:** inspect PV, signed +1bp sensitivity, scenario RMS reduction, key-rate before/after bars, optimized hedge notionals; optionally open synthetic portfolio. Click **Run 70-day rolling / 5-day forward backtest** after loading a sufficiently long history.
- **FedNews commentary:** enter retrieval query; choose local TF-IDF model or no sentiment, inspect note and source-linked headlines; download report JSON.
- **Model & data quality:** inspect test macro-F1, baseline macro-F1, holdout count, confusion matrix, dataset provenance and warnings.

**If dashboard shows 'No public news downloaded'**: run `fetch-news` in another terminal and rerun/reload the dashboard. The Streamlit cache has a 10-minute TTL; restart the server if needed to refresh.

## 10. Expected demo outputs (not real-world findings)

With the supplied fixed synthetic fixture, previously verified on the project code:

| Output | Example value | Meaning |
| --- | ---: | --- |
| Synthetic portfolio PV | $127,592.49 | Swap book mark in simplified curve model |
| Signed parallel +1bp PV change | $13,287.20 per bp | Approximate sensitivity to a small parallel move |
| Scenario RMS BEFORE hedge | $97,282.06 | Scenario-weighted first-order P&L RMS |
| Scenario RMS AFTER hedge | $31,461.00 | Lower *in-sample* synthetic scenario RMS |
| Walk-forward UNHEDGED RMS | $21,612.68 | Future holdout linearized daily P&L RMS |
| Walk-forward 10Y parallel-only RMS | $1,044.00 | Simple comparator, best on this synthetic test |
| Walk-forward OPTIMIZED RMS | $6,814.76 | Better than unhedged, not better than simple baseline |
| Demo sentiment holdout macro-F1 | 0.333 | 9 synthetic test sentences; not a research result |

All values are fixture-dependent and will change with datasets, parameters, and code. To make claims using real data, recompute reports under `--mode real` and label results with exact data sources, dates and limitations.

### Metrics to report in project evaluation

- **Risk:** portfolio PV, signed DV01, key-rate exposure, worst stress P&L, scenario RMS reduction, OOS daily P&L RMS, hedge notional limit hits, and performance versus parallel hedge.
- **Sentiment:** per-class precision/recall/F1, macro-F1, majority-class baseline, confusion matrix, train/test counts; optionally learning curves and validation across domain sources.
- **Retrieval/commentary:** rate-value correctness, future-date-leak checks, citation coverage, retrieval relevance judged by humans, number of missing timestamp-aligned headlines.
- **Systems:** test pass rate, wall-clock CLI runtime, peak RAM, storage size, CPU usage, Streamlit responsiveness. Do not fabricate these measurements.

## 11. Troubleshooting: symptom → cause → exact fix

| Error / symptom | Common cause | Fix |
| --- | --- | --- |
| `py is not recognized` | Windows Python launcher not installed or absent from PATH | Install Python 3.11 from python.org with the launcher; close and reopen terminal; try `python --version` |
| `No suitable Python runtime` / `Requested Python version not installed` | No Python 3.11 available | Install 3.11 or use supported Python 3.10–3.13 and recreate venv |
| `No module named venv` (Linux) | Missing system venv support | Install matching `python3-venv` OS package and recreate environment |
| `Activate.ps1 cannot be loaded` | PowerShell script execution policy | Avoid activation: run `.\.venv\Scripts\python.exe` directly on each command; or use Command Prompt |
| `No module named swapscribe` | Running wrong interpreter or `pip install -e` not run | `python -m pip install -e '.[test]'` inside project root and use that exact Python |
| `No module named streamlit` | Dependencies not installed in selected environment | `python -m pip install -e '.[test]'`, then `python -m streamlit run dashboard.py` |
| `No module named pytest` | Test extra missing | `python -m pip install -e '.[test]'` |
| `pip: command not found` | Wrong invocation/path | `python -m pip ...`, or install pip for your Python |
| `externally-managed-environment` | Installing globally into OS-managed Python | Make/activate a fresh `.venv`; never force-install into system Python |
| `FileNotFoundError: No real market data` | No real Treasury file downloaded | `python -m swapscribe.cli fetch-market --year 2025` |
| `Missing data/real/sentiment_train.csv` | Training before dataset download | `python -m swapscribe.cli fetch-sentiment --agreement 75` then `train --source real` |
| `News missing... fed_news.csv` | RSS not downloaded | Run `fetch-news`; or `commentary --mode demo` |
| `SSLError` / `CERTIFICATE_VERIFY_FAILED` | HTTPS certificate chain, incorrect date or managed network | Check system clock, update trusted CA certificates and `certifi`; try a trusted different network. Do **not** disable TLS verification. |
| `HTTPError 403`, `429` | Official host blocks/rate-limits requests | Retry later, check source website, or manually download from official dataset page and normalize with provided parser |
| `ConnectTimeout` / `ReadTimeout` | Endpoint slow, firewall or internet down | Confirm network, retry; use offline demo or manual official download. Do not assume blank CSV succeeded. |
| `Treasury CSV has no Date column` | Saved HTML webpage instead of CSV, or format changed | Verify downloaded file begins with the CSV headers, then normalize official CSV |
| `No usable Treasury yield observations` | Wrong year/file, parsing errors, or no usable 2Y/10Y columns | Check official CSV content and year; try full 2025 source. |
| `At least three available yield tenors required` | Malformed/partial market data | Ensure valid tenor columns, not all missing. |
| `Walk-forward requires complete node yields` | NaNs in 1,2,3,5,7,10,20Y series | Choose recent full-year Treasury data or clean missing dates using a documented policy (avoid future fill) |
| `Not enough dated history for walk-forward backtest` | Too few daily rates for 70-day lookback and 5-day holding period | Use full-year data or temporarily test `--lookback 30 --holding-days 5` with clear smaller-sample limitation |
| `Need >=7 rate observations` | As-of cutoff too early | Choose a later date |
| `No source headlines...` | No news item dated by cutoff | Download current RSS, use current-year rate file, and select recent as-of date; do not backfill 2025 with 2026 news |
| `Dataset scripts are no longer supported` in Hugging Face | Calling deprecated `datasets.load_dataset` interface directly | Use provided `fetch-sentiment` ZIP downloader, which does not require Hugging Face `datasets` package |
| `ModuleNotFoundError: torch` | Running optional FinBERT without dependencies | Skip it (default baseline needs no Torch) or install `python -m pip install -r requirements-optional-transformers.txt` |
| `Address already in use` / port 8501 busy | Another Streamlit instance running | Stop old server with Ctrl+C or run `python -m streamlit run dashboard.py --server.port 8502` |
| `PermissionError` writing CSV | Excel has file open; read-only folder; controlled folder protection | Close Excel; move project to user-writable local folder; verify permissions |
| `MemoryError` / slow system | Too many apps, optional transformer, excessive dataset | Close heavy applications; use CPU baseline; do not load giant model; reduce simultaneous tasks |
| `joblib` load error / model invalid | Incompatible or untrusted saved model | Retrain locally with `train --source demo` or `train --source real`, using the same environment. Never load unknown pickle files. |
| `pytest` fails unexpectedly | Package version mismatch or running from wrong folder | Run `python diagnose.py`; check working directory, Python version, then recreate isolated venv and reinstall |

**Diagnostic command (new):** `python diagnose.py` prints interpreter location, supported Python version, dependency installation state, and whether each expected data and model file exists. Missing real data is not a failure until you choose real mode.

## 12. Scientific and interview analysis

The guiding research question is: **Can a lightweight, auditable financial-AI workbench jointly reduce selected interest-rate risk measures and generate reproducible market commentary from public data?** Split this into separate falsifiable tasks:

**RQ1 Risk:** Does the constrained 2Y/5Y/10Y hedge reduce in-sample scenario RMS? Does it also reduce the *held-out* daily P&L RMS, and does it beat the parallel-only 10Y comparator? Report actual measured differences and explain mismatched objectives.

**RQ2 NLP:** On held-out Financial PhraseBank examples, does balanced TF-IDF/logistic-regression beat a majority-class dummy in macro-F1? Examine class imbalance and confusion pairs. Random split is not proof of future financial-news generalization.

**RQ3 Grounding:** Does each numerical sentence in the note equal the source rates and date-correct changes? Are all displayed headlines dated before or on the cutoff? Are sources properly linked? This is a more defensible claim than asserting that headlines *caused* price changes.

### Risks / limitations you should volunteer in an interview

1. Treasury par rates are **not** a calibrated SOFR swap curve. Real swap pricing needs instruments, conventions, curve bootstrapping, collateral and multiple projection/discount curves.
2. Portfolio positions are fictional; there is no confidential market or client data.
3. Optimizer regularization is a mathematical position-cost proxy, not measured transaction cost, and +100bp single-node stresses are synthetic.
4. Out-of-sample P&L is first-order rate-shock sensitivity; not full realized coupon-inclusive trade P&L.
5. Financial PhraseBank is sentence-level company/financial sentiment, not gold-standard labels on current Fed press releases.
6. RSS offers recent brief metadata, not full financial-news history. Date-level as-of filtering alone is insufficient for intraday claims.
7. The NLP model is TF-IDF/logistic regression; no RAG vector database or LLM generation is implemented by default.

### Follow-up improvements that are realistic on 16GB RAM

- Add repeated time-window risk backtests across 2024–2026 with a truly chronological train/validation/test design; keep reported 2026 data point-in-time.
- Tune optimizer scenario weighting and costs on a validation period, *not* the final test period.
- Compare TF-IDF classifier with frozen CPU FinBERT inference, measuring accuracy and memory/latency tradeoffs.
- Add more rigorous IR-swap curve construction (e.g., published SOFR OIS references where licensed and available) before claiming pricing precision.
- Add saved experiment configs, `git` commit hashes, package lockfile, and resource monitoring.

## 13. Suggested execution plan (10 study sessions)

1. **Session 1:** Extract project, install Python 3.11 and `venv`, run diagnostic, 19 tests, then the offline `all` command.
2. **Session 2:** Read inputs: data schemas, units, swap trade directions. Understand PV and bp/percent conversions.
3. **Session 3:** Inspect `risk.py`; work one swap PV example by hand; test pay-fixed/receive-fixed sign reversal.
4. **Session 4:** Inspect optimizer inputs, stress scenarios, bounded notionals and synthetic in-sample result.
5. **Session 5:** Run walk-forward tests, explain the parallel-only baseline and why optimizer might underperform it.
6. **Session 6:** Fetch 2025 Treasury data; rerun risk/backtest and record real-year limitations.
7. **Session 7:** Fetch Financial PhraseBank; train real sentiment baseline; inspect per-class confusion matrix.
8. **Session 8:** Fetch current Fed RSS and 2026 yields; produce date-aligned market note; manually verify rates and cited headlines.
9. **Session 9:** Record screenshot/JSON exhibits, hardware performance, and test suite; document reproducibility.
10. **Session 10:** Practice explaining why each step exists, what the data cannot tell us, and the next research improvements.

## 14. Final GitHub and résumé standards

Publish source code, tests, documentation, `pyproject.toml`, and safe synthetic fixtures. Do not publish `.venv/`, downloaded restricted datasets, personal credentials, or pretrained model files without verifying terms. Use the included `.gitignore` and the GitHub Actions test workflow.

**Conservative résumé description:**

> Built a CPU-first financial analytics workbench that evaluates synthetic interest-rate swap portfolios against official U.S. Treasury rate histories, implements constrained hedge optimization with walk-forward baselines, and trains/evaluates an auditable three-class financial sentiment classifier using public, appropriately licensed data.

Only add measured metrics after rerunning on your chosen real-data configuration. Do not claim production pricing, proprietary institutional datasets, or actual trading profits.

## 15. Quick command index

```bash
python diagnose.py
python -m pytest -q
python -m swapscribe.cli --help
python -m swapscribe.cli all
python -m swapscribe.cli fetch-market --year 2025
python -m swapscribe.cli risk --mode real
python -m swapscribe.cli backtest --mode real
python -m swapscribe.cli fetch-sentiment --agreement 75
python -m swapscribe.cli train --source real
python -m swapscribe.cli fetch-market --year 2026
python -m swapscribe.cli fetch-news
python -m swapscribe.cli commentary --mode real --model real --query "inflation interest rate monetary policy"
python -m streamlit run dashboard.py
```

Read the existing `docs/METHODOLOGY.md`, `docs/ARCHITECTURE.md`, and `docs/INTERVIEW_GUIDE.md` for the in-depth assumptions and role-focused discussion prompts.
