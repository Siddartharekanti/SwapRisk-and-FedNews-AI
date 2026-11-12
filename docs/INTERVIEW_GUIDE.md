# BNP Paribas Global Markets Data & AI Lab — project interview guide

## 1. Explain your project in 30 seconds

> I built a CPU-first global-markets research workbench combining two independent pipelines. The first values an illustrative fixed-for-floating swap book using a documented Treasury-based teaching curve, calculates parallel and key-rate exposure, and optimizes bounded reference hedges. It evaluates those hedges not only in sample but on strictly future rate moves. The second ingests official public finance news metadata, trains a compact financial sentiment classifier, retrieves relevant pre-date-cutoff headlines, and generates numerically auditable market commentary. The application works on a 16 GB RAM laptop and makes data provenance and modeling limitations explicit.

## 2. Why this project is relevant

The specified internship mentions (a) interest-rate swap risk optimization, (b) automated market comments, (c) hypothesis testing and statistics, (d) extracting insights from diverse data, (e) Python/scikit-learn/PyTorch/Hugging Face, and (f) explaining why models behave as they do.

## 3. Questions to rehearse

1. **What is a plain vanilla interest-rate swap?** A contract exchanging fixed and floating interest cashflows on a notional; the principal itself is normally not exchanged.
2. **What is DV01?** Dollar present-value sensitivity to a 1-basis-point change. Our measure uses a signed finite-difference convention, which is documented in the output.
3. **Why key-rate DV01 rather than only parallel DV01?** Non-parallel curve twists can cause losses even when net parallel exposure is near zero.
4. **Why optimize instead of manually matching DV01?** Optimization can consider multiple nodes, scenarios, bounds and regularization together.
5. **What guarantees does the optimizer have?** Bounded linear least squares is convex; the best solution for the *specified linearized objective* is not guaranteed to minimize real-world hedging losses.
6. **What data is real?** Treasury public market yields and downloaded Federal Reserve metadata; swaps and demo headlines are synthetic. Financial PhraseBank is a separate licensed sentence dataset.
7. **Why not call Treasury par yields SOFR zero rates?** They are economically and mathematically distinct. Par bond yields do not represent the OIS discount curve needed for market-consistent SOFR swaps.
8. **Which pricing assumptions are missing?** Actual SOFR fixings, OIS bootstrapping, overnight compounding, business calendars, cashflow accruals, convexity, valuation adjustments, and real trading costs.
9. **Where is future-data leakage prevented?** The backtest restricts training shocks to dates at or before each rebalance; the news retriever filters headline publication dates relative to as-of.
10. **Why TF-IDF + logistic regression rather than a huge transformer?** It is trainable and explainable locally, gives a strong low-cost baseline, and makes ablation and model comparisons practical.
11. **How do you know an NLP model works?** Stratified held-out macro-F1, class confusion matrix, dummy baseline, then an external dated test set if available.
12. **Why macro-F1?** Gives each class equal weight, preventing the majority class from dominating the score.
13. **Is the demo model benchmark meaningful?** No; 36 fabricated sentences test pipeline integration, not financial generalization. The real model must be trained on the downloaded licensed benchmark and evaluated separately.
14. **How does your system avoid hallucinations?** All quantitative commentary is computed from the input table. News references come only from retrieved rows, with date/source/URL. No free-form factual LLM generation occurs.
15. **Can you say the news caused the yield movement?** No; headline retrieval shows context, not identified causation. An event study with proper timestamps and controls would be required.
16. **What is the most important ablation?** Compare optimized hedge with unhedged and parallel-only out of sample; compare financial NLP with most-frequent dummy and optionally FinBERT on a labelled holdout.
17. **What would you build next with institutional market data?** Market-standard multi-curve SOFR pricing, quotes, calibration checks, transaction cost and liquidity constraints, scenario controls, and monitored production pipeline.
18. **Why are 16 GB RAM and 3 GB GPU sufficient?** Sparse TF-IDF matrices and small yield data fit in CPU memory; constrained least squares solves small matrix systems; the dashboard does not train large LLMs.
19. **What is your biggest project limitation?** Using Treasury par yields as direct discount-rate proxies. It is transparent for a portfolio demo but not suitable for production swap valuations.
20. **What did you learn from the statistical results?** Discuss actual regenerated `artifacts/backtest_real.json` and `sentiment_real_eval.json` only after collecting public data; never pass off demo metrics as measured real-world outcomes.

## 4. Suggested honest résumé bullets (edit only after verifying results)

- Developed a reproducible Python financial risk demonstrator with fixed/floating swap PV, multi-tenor key-rate sensitivity, bounded hedge optimization and a walk-forward test against two baselines.
- Engineered a financial-news NLP pipeline using a TF-IDF/logistic-regression classifier, time-filtered evidence retrieval and mechanically verified market commentary, exposed through a local Streamlit dashboard.
- Designed an audit-friendly research workflow with explicit dataset licensing, source provenance, unit/integration tests, data-leakage guards and reproducible model evaluation.

**Do not insert numeric improvements into the résumé until you have genuinely run and documented the result on named real-data periods.**
