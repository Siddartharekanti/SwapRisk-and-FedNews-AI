# Real datasets and provenance

This repository contains **ONLY SYNTHETIC** demo data. The following **public data importers** run on your machine; the ZIP does **not** include or pretend to include proprietary financial data or copied licensed data.

## 1. Official U.S. Treasury daily par yield curve

- Documentation: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve
- Source description: https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/
- Year download pattern (`market.py`):
  `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv`
- Dates and columns: date; annual percent yields for 3 months, 6 months, 1/2/3/5/7/10/20/30 years.
- `python -m swapscribe.cli fetch-market --year 2025`
- **Limit:** These are Treasury government security *par yields*, not USD SOFR/OIS swap rates. The swap module deliberately uses them as approximated zero rates for a teaching-only proxy. In general this conversion is not valid for market pricing.

## 2. Federal Reserve Board press releases (RSS)

- Directory: https://www.federalreserve.gov/feeds/feeds.htm
- RSS source: https://www.federalreserve.gov/feeds/press_all.xml
- `python -m swapscribe.cli fetch-news`
- Extracts headline, short description, publication date, URL, source. This is **not** a large historical full-text news dataset and does not itself contain returns, spreads, or swap price quotes. The latest feed may contain unrelated administrative releases; query relevance filtering helps but won't eliminate that limitation.
- To build a historical causal study, obtain a properly licensed historical event database with reliable publication timestamps.

## 3. Financial PhraseBank

- Dataset card: https://huggingface.co/datasets/takala/financial_phrasebank
- Dataset ZIP: https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip
- Research paper: Malo et al., *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts*, 2014 (see dataset card).
- License on the HF dataset card: **CC BY-NC-SA 3.0** — attribution, noncommercial, share alike.
- `python -m swapscribe.cli fetch-sentiment --agreement 75`
- The model uses the 75%-agreement subset; labels are positive/neutral/negative.
- Dataset lacks genuine release timestamps and predefined train/validation/test split. We do a seed-fixed stratified holdout on deduplicated sentences. That is a **sentence-sentiment benchmark**, not an out-of-time test on current Fed press releases.
- Do not redistribute the downloaded content or use it commercially without checking the license and authorization.

## 4. Optional pretrained FinBERT model

- https://huggingface.co/ProsusAI/finbert
- Optional `pip install -r requirements-optional-transformers.txt`, then call `finbert_cpu(news_frame)` from `swapscribe.sentiment`.
- This downloads a transformer checkpoint on first use and runs inference **on CPU**, so it does not require the user's 3 GB VRAM.
- Different models may have different license terms; review the model card before reuse.

## Quality control

1. Never mix demo rows with real rows silently.
2. Never pretend official Treasury yields are transaction-level swap prices.
3. Never impute future-published news into historical days.
4. Preserve and report dataset source, observation dates and comments on missing fields.
5. Public data does not establish client-level relevance; no private banking-client data is included.
