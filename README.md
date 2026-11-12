# SwapRisk-and-FedNews-AI

**Quantitative Interest-Rate Swap Risk Engine & Grounded Federal Reserve Macro Sentiment Architecture**

[![CI Test Suite](https://github.com/Siddartharekanti/SwapRisk-and-FedNews-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Siddartharekanti/SwapRisk-and-FedNews-AI/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)](tests/)

An institutional-grade, publication-ready quantitative finance and macro natural language processing research framework. The system implements exact par-to-zero yield curve bootstrapping, multi-tenor interest rate swap valuation, key-rate Greeks (first-order DV01 and second-order Gamma convexity), constrained convex quadratic hedge optimization, walk-forward out-of-sample backtesting with transaction friction, and leakage-controlled Federal Reserve communication intelligence.

---

## 1. Abstract & Key Contributions

Interest rate risk management across multi-tenor institutional swap portfolios requires addressing non-parallel curve dynamics (twist, steepening, flattening) while managing turnover and liquidity constraints. Concurrently, integrating macro sentiment signals from central bank communications demands strict temporal guards to eliminate lookahead bias and hallucinated causality.

**SwapRisk-and-FedNews-AI** delivers an end-to-end reproducible architecture:
1. **Arbitrage-Free Bootstrapping**: Inductively extracts continuous zero rates $z(t)$ and discount factors $D(t)$ from official Treasury par yields.
2. **Key-Rate Greeks & Second-Order Convexity**: Computes full 10-node DV01 sensitivity profiles and explicit key-rate Gamma ($\Gamma_k$) matrices to capture non-linear rate shocks.
3. **Constrained Quadratic Convex Optimizer**: Solves for bounded reference hedge allocations (2Y, 5Y, 10Y liquid benchmarks) via SLSQP with $L_2$ regularization.
4. **Leakage-Controlled Walk-Forward Backtester**: Implements rolling out-of-sample holding intervals with 0.5 bp bid-ask transaction friction.
5. **Class-Balanced Macro Sentiment AI**: Evaluates Financial PhraseBank (75% agreement) with sublinear TF-IDF and balanced multinomial logistic regression, achieving **85.15% accuracy** and **0.8094 Macro-F1** (+217% over majority-class baseline).
6. **Temporal Grounded Retrieval**: Eliminates lookahead bias with strict $t \le \text{as-of}$ date filtering and auditable citation provenance.

---

## 2. Empirical Benchmark Performance (Real 2026 Data)

Evaluated on official **2026 U.S. Treasury daily par yields**, **Federal Reserve RSS metadata**, and **Financial PhraseBank** (3,448 samples):

| Metric Category | Specific Indicator | Baseline / Unhedged | SwapRisk-and-FedNews-AI | Empirical Lift |
| :--- | :--- | :---: | :---: | :---: |
| **Portfolio Valuation** | Net Present Value (PV) | — | **+$843,877.12** | Calibrated at par |
| **Duration Sensitivity** | Parallel DV01 (+1 bp) | — | **+$12,392.19 / bp** | Fixed receiver bias |
| **Stress Scenario Risk** | RMS Scenario P&L | $114,183.68 | **$45,684.83** | **-60.0% variance reduction** |
| **Tail Risk Drawdown** | Worst Stress Scenario | -$960,951.57 | **-$311,887.92** | **-67.5% drawdown reduction** |
| **Out-of-Sample Backtest**| 140-day OOS RMS P&L | $59,501.06 | **$24,950.55** | **-58.1% volatility reduction** |
| **Turnover & Friction** | Net of 0.5 bp Friction | — | **$24,946.69** | \$1,167.80 total friction |
| **Sentiment Accuracy** | 862-sample holdout test | 62.06% (majority) | **85.15%** | **+23.1% absolute** |
| **Sentiment Macro-F1**| Stratified 3-class F1 | 0.2553 (majority) | **0.8094** | **+217.0% relative** |

---

## 3. Mathematical Core

### Exact Par-to-Zero Bootstrapping
Given semiannual par yields $y_n$ at maturities $t_n$, discount factors are inductively derived:
$$D(t_n) = \frac{1 - \frac{y_n}{2}\sum_{i=1}^{n-1} D(t_i)}{1 + \frac{y_n}{2}}, \quad z(t_n) = -\frac{\ln D(t_n)}{t_n}$$

### Second-Order Greeks & P&L Expansion
Taylor series expansion incorporating first-order DV01 ($S_k$) and second-order Gamma convexity ($\Gamma_k$):
$$\Delta PV \approx \sum_{k=1}^K S_k \Delta y_k + \frac{1}{2} \sum_{k=1}^K \Gamma_k (\Delta y_k)^2$$
where $\Gamma_k = \frac{PV(y + \delta e_k) - 2PV(y) + PV(y - \delta e_k)}{\delta^2}$.

### Multi-Objective Convex Optimization
$$\min_{h \in \mathbb{R}^M} \frac{1}{P} \| S (r + H h) \|_2^2 + \lambda \| h \|_2^2 \quad \text{subject to} \quad -h_{\max} \le h_j \le h_{\max}$$
solved via SLSQP with box constraints ($h_{\max} = \$30\text{MM}$) and $L_2$ regularization ($\lambda = 200$).

---

## 4. Quickstart & Replication

### 4.1 Installation

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

### 4.2 System Diagnostic & Test Suite

```bash
# Verify environment and dependencies
python diagnose.py

# Run 19 mathematical & integration tests
python -m pytest -q
```

### 4.3 End-to-End Real-Data Pipeline

```bash
# 1. Fetch public datasets (official U.S. Treasury, Fed RSS, Financial PhraseBank)
python -m swapscribe.cli fetch-market --year 2026
python -m swapscribe.cli fetch-news
python -m swapscribe.cli fetch-sentiment --agreement 75

# 2. Train sentiment model and run full quant workflow
python -m swapscribe.cli all --mode real

# 3. Launch interactive Streamlit research dashboard
python -m streamlit run dashboard.py
```

The interactive dashboard opens at `http://localhost:8501`.

---

## 5. Repository Structure

```text
SwapRisk-and-FedNews-AI/
├── dashboard.py                  # Interactive Streamlit quantitative research dashboard
├── diagnose.py                   # Local environment and data diagnostic utility
├── pyproject.toml                # Package configuration (swaprisk-and-fednews-ai v1.0.0)
├── requirements.txt              # Production dependency lockfile
├── src/
│   └── swapscribe/
│       ├── cli.py                # Command-line interface and orchestrator
│       ├── market.py             # Par-to-zero bootstrapping, curve construction, market loaders
│       ├── risk.py               # Swap pricing, key-rate DV01 Greeks, stress scenarios
│       ├── backtest.py           # Walk-forward backtesting with Gamma convexity & friction
│       ├── optimizer.py          # Bounded SLSQP quadratic convex hedge optimizer
│       ├── sentiment.py          # TF-IDF + balanced logistic regression classifier
│       ├── commentary.py         # Grounded temporal retrieval and market note generation
│       ├── data_sources.py       # Importers for Treasury, Fed RSS, and PhraseBank
│       └── paths.py              # Canonical project filesystem paths
├── tests/                        # 19 unit & mathematical validation tests
├── docs/
│   ├── ARCHITECTURE.md           # System architecture and dataflow specifications
│   ├── DATA_SOURCES.md           # Data provenance and license compliance documentation
│   ├── METHODOLOGY.md            # Detailed quantitative methods, equations, and baselines
│   └── COMPLETE_SETUP_TROUBLESHOOTING_AND_ANALYSIS.md # Full technical specification & runbook
└── artifacts/                    # Persisted model weights and evaluation JSON outputs
```

---

## 6. Research Documentation

- **[System Architecture](docs/ARCHITECTURE.md)**: Dataflow pipelines, input/output schemas, and caching layers.
- **[Empirical Methodology](docs/METHODOLOGY.md)**: Mathematical formulations, pricing identities, and baseline comparisons.
- **[Data Provenance & Licensing](docs/DATA_SOURCES.md)**: Public source endpoints, licensing guidelines, and temporal leakage guards.
- **[Technical Specification & Replication Guide](docs/COMPLETE_SETUP_TROUBLESHOOTING_AND_ANALYSIS.md)**: Detailed execution guide and troubleshooting matrix.

---

## 7. Citation & License

This project is licensed under the [MIT License](LICENSE).

```bibtex
@software{swaprisk_fednews_ai_2026,
  author = {Siddartha Rekanti},
  title = {SwapRisk-and-FedNews-AI: Quantitative Interest-Rate Swap Risk Engine and Grounded Federal Reserve Macro Sentiment Architecture},
  year = {2026},
  url = {https://github.com/Siddartharekanti/SwapRisk-and-FedNews-AI}
}
```
