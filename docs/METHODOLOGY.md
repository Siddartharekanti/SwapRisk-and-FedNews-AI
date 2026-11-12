# Methods, equations, baselines, experiments and limitations

## 1. Main research questions

**Q1, rates:** Can a bounded portfolio of 2Y/5Y/10Y pay-fixed reference swaps reduce scenario-based rate exposure for a synthetic book, compared with (a) doing nothing and (b) hedging only parallel DV01 using a single 10Y hedge?

**Q2, NLP:** Does a sparse TF-IDF + balanced logistic-regression classifier outperform a majority-class baseline for three-way financial sentence sentiment on held-out Financial PhraseBank records?

**Q3, retrieval:** Can a date-filtered retrieval pipeline produce a numerical market summary with all quantitative claims traceable to observable rates and all editorial context traceable to dated, attributed headlines?

## 2. SwapRisk-and-FedNews-AI: Swap Risk & Hedging Mathematics

### Instrument definition

- `Swap(swap_id, notional_usd, maturity_years, fixed_rate_pct, direction)`
- Positive `pay_fixed` receives the floating leg and pays fixed.
- `receive_fixed` reverses the sign.
- Payments every 0.5 years (`delta=0.5`). This deliberately omits actual holiday calendars, Act/360/30/360 conventions, compounding, accrued cashflows, stubs, and collateral.

The illustrative discount factor is:

`D(t) = exp[-z_proxy(t) * t]`

where `z_proxy(t)` is obtained by piecewise-linear interpolation of reported Treasury **par** yields (percent divided by 100), erroneously treating them as continuous zero rates from a pure instrument-pricing perspective. This is a **teaching simplification** not valid for market-consistent swap valuation.

Fixed annuity: `A(T) = sum_{i=1..2T} 0.5 * D(i/2)`.

Floating PV at reset **in an ideal single-curve model**: `PV_float = N * (1 - D(T))`.

Fixed PV: `PV_fixed = N * k * A(T)` where `k=fixed_rate_pct/100`.

For pay-fixed:

`PV = PV_float - PV_fixed`

Par rate: `k_par = (1 - D(T)) / A(T)`.

### Risk measures

- Signed 1bp parallel sensitivity: `PV(z+0.5bp) - PV(z-0.5bp)`.
- Key-rate node sensitivity for tenor `t`: `PV(z_t+0.5bp) - PV(z_t-0.5bp)`.
- Nodes: **1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y**.
- These risk factors are **not** full sensitivity to all Treasury nodes. In particular the 3M and 6M nodes are not separately shocked.
- Finite differences are local. ±100bp full repricing is also reported for the unhedged portfolio to show nonlinear effects.

### Scenarios and optimization

Take the past `min(260, available−1)` successive daily changes in U.S. Treasury yields (percent * 100 = basis points). Append **±100 bp at each key node**, one-at-a-time, as deterministic stress cases. Each scenario is a row of `S` in bp.

Let `r` be the current synthetic portfolio key-rate sensitivity (USD per bp), `H` be the matrix of unit-$1MM hedge sensitivities for available at-par 2Y/5Y/10Y swaps, and `x` be signed USD million hedge amounts. Solve:

`min_x  ||S (r + Hx)||² / n + λ ||x||²`

subject to `-L <= x_i <= L`.

Here `L` defaults to `$30MM` per reference instrument and `λ=200` is a *dimensioned L2 position regularizer*. `scipy.optimize.lsq_linear` solves the stacked bounded linear least-squares system. Positive x: pay fixed. Negative x: receive fixed. Neither `λ` nor the hedge constraints are calibrated from actual trading costs or liquidity.

Reporting: scenario root-mean-square (RMS) approximation, worst linearized scenario P&L, signed key-rate sensitivity, swap PV, and full repriced ±100bp unhedged portfolio P&L. Scenario weights are equal and stress cases mix with historical shocks by design, which may overweight stresses relative to their frequencies.

### Out-of-sample walk-forward

For a date `t`, construct optimizer features **only from the preceding `lookback` changes up to and including `t`**. Optimize weights at t, then apply the frozen sensitivities and hedge weights to the **next** `holding_days` daily yield changes. Rebalance at the following interval. No future rate changes enter t's hedge fit.

Compare OOS first-order daily P&L RMS of:

- Unhedged synthetic swap book;
- Single at-par 10Y hedge sized to cancel aggregate signed key-rate sensitivity (parallel-only baseline), bounded to the same maximum USD notional;
- Optimized 2Y/5Y/10Y hedge.

OOS estimates are still **linearized** rather than actual trade P&L; they exclude coupons, transaction costs, carry, funding and intraday risks. Improvement is **not guaranteed** on new real-data windows. More elaborate research should add financing/coupon cashflows and reprice all swaps with market-standard curves.

## 3. FedNews Macro Sentiment & Attribution

### Dataset

Financial PhraseBank 75% agreement: approx 3,453 labelled sentences in three classes. The importer stores only one agreement subset and removes duplicate normalized sentences. The benchmark's license is CC BY-NC-SA 3.0; downloaded bytes are not redistributed with this repository.

### Model and evaluation

- `TfidfVectorizer(ngram_range=(1,2), max_features=12000, sublinear_tf=True)`;
- `LogisticRegression(class_weight='balanced', max_iter=700)`;
- Deterministic `train_test_split(test_size=.25, random_state=42, stratify=label)`;
- Heldout macro-F1, accuracy and full confusion matrix;
- `DummyClassifier(strategy='most_frequent')` baseline trained on same training fold.

The project **does not** claim a temporal sentiment evaluation, because the dataset has no article publication timestamps and no official forward split. No hyperparameter tuning on the test set is performed. The demo training set is only 36 synthetic sentences and its numerical score is not meaningful for real-world performance.

### Optional transformer comparison

With `requirements-optional-transformers.txt`, use `sentiment.finbert_cpu(news_frame)`. It loads `ProsusAI/finbert` on **CPU** with `batch_size=4`, avoiding the 3 GB GPU entirely. Its predictions can be compared with TF-IDF on a properly labelled holdout; do not claim improved performance without running that comparison.

## 4. Retrieval and market note

- Query and publication-date-filtered news rows are TF-IDF vectorized by `TfidfVectorizer(ngram_range=(1,2), stop_words='english')`;
- Cosine similarity ranks candidates; recency breaks ties.
- Only metadata dated on/before as-of appears, preventing future-record leakage at daily resolution.
- The 2Y and 10Y yields and 10Y–2Y spread are computed directly from Treasury rate columns; day-on-day change uses prior **available observation** (not necessarily prior calendar day).
- The output has a dated note and references with source URL, date, lexical relevance and (optionally) model sentiment and uncalibrated probability.
- It explicitly says related headlines are contextual; **no causal explanation** is inferred from correlations or sentiment.
- Unlike an open-ended LLM, the template does not invent numbers, articles or narrative explanations. Model interpretability and defensibility matter more than a demo that appears fluent but hallucinates.

## 5. Data-safety and leakage checklist

1. Demo market and portfolio are marked synthetic in CLI and Streamlit.
2. Downloaders emit exceptions rather than silently replacing real data with demo values.
3. Different modes have different stored files, separate sentiment models and different evaluation JSONs.
4. Old as-of selection truncates future rate observations and future news.
5. No user proprietary data, credentials, or real client identifiers are needed.
6. Model files use joblib/pickle, so deserialize only trusted artifacts created on the user's own machine.
7. No claim that an RSS item's publication date proves its release preceded same-day market close.
8. No claim of calibrated OIS / risk-neutral curve valuation.

## 6. Next-level extension for stronger interviews

To turn the educational demonstrator into a serious quant project, source **historical SOFR OIS swap quotes** under appropriate data rights, bootstrap discount and projection curves using QuantLib, add USD swap conventions with Business/252 vs Act/360 checks as appropriate, implement cashflow schedules/stubs, use adjoint risk, add liquidity costs and actual out-of-sample realized revaluation. Document results against an open-source benchmark, then build a simple experiment tracker (git commit, data hash, random seed, hyperparameters).
