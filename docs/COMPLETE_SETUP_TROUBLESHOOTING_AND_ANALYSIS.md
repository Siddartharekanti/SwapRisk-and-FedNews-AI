# Technical Specification, Empirical Replication & Architecture Guide

An institutional-grade technical specification and mathematical runbook for **SwapRisk-and-FedNews-AI** — a research architecture combining interest-rate swap risk optimization, key-rate Greeks, second-order convexity backtesting, and grounded Federal Reserve communication intelligence.

---

## 1. Executive Summary & Research Framework

The platform addresses two fundamental challenges in computational finance and macro intelligence:

1. **Multi-Tenor Swap Book Risk Minimization**: Constructing an exact, arbitrage-free yield discount curve from benchmark par yields, projecting first-order key-rate DV01 and second-order Gamma convexity across non-parallel curve shifts, and solving a constrained convex quadratic optimization problem to determine minimum-variance hedge notionals under turnover friction.
2. **Leakage-Controlled Central Bank Intelligence**: Eliminating forward lookahead bias in natural language pipelines by strictly conditioning news retrieval on point-in-time publication dates, classifying sentiment via a class-balanced probabilistic model, and anchoring all quantitative market claims to observable yield differences without hallucinating causal narratives.

```text
                           [Observable Market Data]
                     /                                  \
          U.S. Treasury Par Yields              Federal Reserve Press RSS
                     |                                       |
          Exact Par-to-Zero Bootstrapping        Point-in-Time Date Filtering
                     |                                       |
          Multi-Tenor Discount Curve               Lexical Cosine Retrieval
                     |                                       |
          Fixed/Floating Swap Valuation          Attributed Headline Provenance
                     |                                       |
          Key-Rate DV01 & Gamma Greeks                       |
                     |                                       |
          Convex Quadratic Optimization                      |
                     |                                       |
          Walk-Forward OOS Backtesting                       |
                     \                                       /
                      [Streamlit Institutional Dashboard]
```

---

## 2. Quantitative Mathematical Formulation

### 2.1 Exact Par-to-Zero Curve Bootstrapping
Let $y_n$ denote the observed par yield for benchmark tenor $t_n$ with semiannual fixed coupon payments ($\delta = 0.5$). The par bond identity dictates:
$$1 = \sum_{i=1}^{n-1} \frac{y_n}{2} D(t_i) + \left(1 + \frac{y_n}{2}\right) D(t_n)$$

Solving inductively for the unknown discount factor $D(t_n)$:
$$D(t_n) = \frac{1 - \frac{y_n}{2}\sum_{i=1}^{n-1} D(t_i)}{1 + \frac{y_n}{2}}$$

The corresponding continuously compounded zero rate $z(t_n)$ is:
$$z(t_n) = -\frac{\ln D(t_n)}{t_n}$$

Continuous zero rates between benchmark tenors are obtained via linear interpolation on $z(t)$, ensuring monotonic discount factors $D(t) = \exp(-z(t) \cdot t)$.

### 2.2 Swap Valuation & Greeks
For a fixed-for-floating interest rate swap with notional $N$, fixed coupon $k$, and maturity $T$:
- **Fixed Leg Annuity**: $A(T) = \sum_{i=1}^{2T} \delta \cdot D(t_i)$
- **Floating Leg PV**: $PV_{\text{float}} = N \cdot (1 - D(T))$
- **Fixed Leg PV**: $PV_{\text{fixed}} = N \cdot k \cdot A(T)$
- **Net Swap PV (Pay-Fixed)**:
  $$PV = PV_{\text{float}} - PV_{\text{fixed}} = N \left[ (1 - D(T)) - k A(T) \right]$$
  *(Receive-fixed swaps negate the net present value).*

#### First-Order Greeks (Key-Rate DV01)
Vector sensitivities across tenors $t_k \in \{1\text{Y}, 2\text{Y}, 3\text{Y}, 5\text{Y}, 7\text{Y}, 10\text{Y}, 20\text{Y}\}$:
$$S_k = \frac{PV(y + \delta e_k) - PV(y - \delta e_k)}{2\delta}, \quad \delta = 0.5 \text{ bp}$$

#### Second-Order Greeks (Key-Rate Convexity / Gamma)
Key-rate curvature capturing non-linear curve shifts:
$$\Gamma_k = \frac{PV(y + \delta e_k) - 2PV(y) + PV(y - \delta e_k)}{\delta^2}$$

### 2.3 Constrained Convex Quadratic Optimization
Let $r \in \mathbb{R}^K$ represent the unhedged portfolio key-rate sensitivity vector. Let $H \in \mathbb{R}^{K \times M}$ denote the sensitivity matrix of $M$ liquid benchmark hedging instruments (2Y, 5Y, 10Y at-par swaps) per $\$1\text{MM}$ notional. Under historical curve scenarios and deterministic stress shocks $S \in \mathbb{R}^{P \times K}$:

$$\min_{h \in \mathbb{R}^M} \frac{1}{P} \| S (r + H h) \|_2^2 + \lambda \| h \|_2^2 \quad \text{subject to} \quad -h_{\max} \le h_j \le h_{\max}$$

where:
- $\lambda = 200$ is an $L_2$ regularization penalty preventing excessive leverage and concentration.
- $h_{\max} = \$30\text{MM}$ imposes realistic counterparty and execution liquidity limits.
- Solved via bounded linear least-squares (`scipy.optimize.lsq_linear` / SLSQP).

### 2.4 Out-of-Sample Walk-Forward Backtesting
To prevent lookahead bias:
1. At estimation date $t$, the optimizer fits weights $h(t)$ using **only historical yield changes up to $t$**.
2. Portfolio sensitivity $r(t)$ and hedge allocations $h(t)$ are frozen and evaluated over the forward holding interval $\tau \in [t+1, t+H]$:
   $$\Delta PV_\tau \approx \sum_k S_k(t) \Delta y_{k,\tau} + \frac{1}{2} \sum_k \Gamma_k(t) (\Delta y_{k,\tau})^2 - \text{Friction}_\tau$$
3. **Transaction Costs**: Portfolio rebalancing incorporates a 0.5 bp bid-ask turnover friction:
   $$\text{Friction}_\tau = \sum_{j=1}^M |h_j(t) - h_j(t-1)| \times 5 \times 10^{-5}$$

---

## 3. Macro NLP & Grounded Attribution Architecture

### 3.1 Financial Sentiment Classification
- **Benchmark**: Financial PhraseBank (75% annotator agreement, 3,448 unique financial sentences).
- **Feature Pipeline**: Sublinear TF-IDF with $(1, 2)$-grams (max 12,000 features), directional stopword filtering, and negation prefix parsing.
- **Classifier**: Multinomial Logistic Regression with inverse class-frequency balancing:
  $$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
- **Validation**: Seed-fixed stratified holdout (75/25 split). Evaluated via Macro-averaged $F_1$, Precision, Recall, and Confusion Matrix against a naive majority-class baseline.

### 3.2 Temporal Lexical Retrieval & Grounding
- **Observation Filtering**: Strictly enforces $\text{Date}(\text{Headline}) \le \text{Date}_{\text{as-of}}$, guaranteeing zero forward information leakage.
- **Attribution Guard**: Rates and spreads are computed analytically from observable curve columns. Retrieved headlines provide macroeconomic context and are explicitly marked as non-causal.

---

## 4. Empirical Benchmark Results (Real 2026 Public Data)

Replication executed on **official 2026 U.S. Treasury daily par yields**, **Federal Reserve RSS metadata**, and **Financial PhraseBank 75% agreement**:

| Metric Category | Specific Indicator | Baseline / Unhedged | SwapRisk-and-FedNews-AI | Empirical Lift |
| :--- | :--- | :---: | :---: | :---: |
| **Portfolio Valuation** | Net Present Value (PV) | — | **+$843,877.12** | Calibrated at par |
| **Duration Sensitivity** | Parallel DV01 (+1 bp) | — | **+$12,392.19 / bp** | Fixed receiver bias |
| **Stress Risk** | RMS Scenario P&L | $114,183.68 | **$45,684.83** | **-60.0% variance** |
| **Tail Risk** | Worst Stress Drawdown | -$960,951.57 | **-$311,887.92** | **-67.5% drawdown** |
| **OOS Backtesting** | 140-day OOS RMS P&L | $59,501.06 | **$24,950.55** | **-58.1% volatility** |
| **Turnover & Cost** | Net of 0.5 bp Friction | — | **$24,946.69** | \$1,167.80 total cost |
| **Sentiment Accuracy** | 862-sample holdout test | 62.06% (majority) | **85.15%** | **+23.1% absolute** |
| **Sentiment Macro-F1**| Stratified 3-class F1 | 0.2553 (majority) | **0.8094** | **+217.0% relative** |

---

## 5. Execution & Replication Runbook

### 5.1 Environment Setup

```powershell
# 1. Initialize virtual environment (Python 3.11 recommended)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install package in editable mode
python -m pip install --upgrade pip
python -m pip install -e ".[test]"

# 3. Verify local environment and integrity
python diagnose.py
python -m pytest -q
```

### 5.2 End-to-End Real Pipeline Execution

```powershell
# 1. Download official U.S. Treasury par yield curve data
python -m swapscribe.cli fetch-market --year 2025
python -m swapscribe.cli fetch-market --year 2026

# 2. Ingest official Federal Reserve RSS feed
python -m swapscribe.cli fetch-news

# 3. Ingest Financial PhraseBank dataset (75% agreement)
python -m swapscribe.cli fetch-sentiment --agreement 75

# 4. Train and validate sentiment classifier
python -m swapscribe.cli train --source real

# 5. Execute full real-data workflow (Valuation, Optimization, Backtest, Commentary)
python -m swapscribe.cli all --mode real

# 6. Launch interactive dashboard
python -m streamlit run dashboard.py
```

---

## 6. Output Artifacts & Evidence Files

Every execution persists structured JSON artifacts ready for downstream analysis:

- `artifacts/risk_real.json`: Net present value, parallel DV01, key-rate sensitivities, optimal hedge notionals, and scenario distributions.
- `artifacts/backtest_real.json`: Walk-forward out-of-sample daily P&L series, turnover tracking, transaction costs, and comparative RMS tables.
- `artifacts/sentiment_real_eval.json`: Test set classification report, holdout confusion matrix, macro-F1, and majority-class baseline.
- `artifacts/commentary_real.json`: Date-stamped factual yield-curve commentary with attributed Fed RSS source URLs.

---

## 7. Diagnostics & Troubleshooting Matrix

| Symptom | Primary Cause | Resolution |
| :--- | :--- | :--- |
| `FileNotFoundError: No real market data` | Market files missing from `data/real/` | Execute `python -m swapscribe.cli fetch-market --year 2026`. |
| `Missing data/real/sentiment_train.csv` | Sentiment data not downloaded | Execute `python -m swapscribe.cli fetch-sentiment --agreement 75`. |
| `News missing... fed_news.csv` | RSS feed not fetched | Execute `python -m swapscribe.cli fetch-news`. |
| `Address already in use: 8501` | Existing Streamlit process active | Stop prior process or run with `--server.port 8502`. |
| `SSLError / CERTIFICATE_VERIFY_FAILED` | Local machine CA certificate expired | Update `certifi` package via `python -m pip install --upgrade certifi`. |
