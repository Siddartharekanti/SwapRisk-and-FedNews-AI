# System architecture

```text
                +-----------------------------+
                |   External public sources   |
                | Treasury CSV / Fed RSS / FPB|
                +--------------+--------------+
                               |
                     EXPLICIT DOWNLOAD
                               |
             +-----------------+-----------------+
             |                                   |
      Data/real CSVs                         Data/demo CSVs
      (public source)                       (SYNTHETIC ONLY)
             |                                   |
             +-----------------+-----------------+
                               |
                    Input source selection
                               |
                   +-----------+-----------+
                   |                       |
         SwapShield-Q market/risk     MacroScribe NLP
                   |                       |
          curve proxy + portfolio     TF-IDF sentiment
                   |                       |
         PV / DV01 / key rates       lexical retrieval
                   |                       |
          hedge scenario fitting     as-of source filter
                   |                       |
        walk-forward OOS testing     factual template note
                   |                       |
                   +-----------+-----------+
                               |
                    JSON evidence artifacts
                               |
                     Streamlit dashboard
```

## Modeling assumptions

- Financial data is kept in its original form with `date` and source provenance.
- Market data uses a *single-curve*, continuous zero-rate approximation **derived from par Treasury yields**. It is a teaching proxy only.
- The swap portfolio consists solely of synthetic USD trades, even under real-market mode.
- Each headline is attributed to an explicit dated source. RSS yields recent source metadata only.
- The sentiment pipeline avoids heavyweight model training by using CPU-only sparse text features.
- The entire app starts offline with `python -m swapscribe.cli all` after packages are installed.

## Interfaces

| Interface | Input | Output |
| --- | --- | --- |
| `fetch-market` | year | `data/real/treasury_YEAR.csv` |
| `fetch-news` | Fed RSS | `data/real/fed_news.csv` |
| `fetch-sentiment` | FPB zip | `data/real/sentiment_train.csv` |
| `train` | `sentiment_train.csv` | `.joblib` model, eval JSON |
| `risk` | yield CSV, synthetic trades | risk JSON |
| `backtest` | historical curve CSV, synthetic trades | OOS JSON |
| `commentary` | as-of yields + RSS + optional sentiment | cited note JSON |
| `dashboard.py` | same files | interactive local browser UI |

## Further engineering ideas

- Model run registry with dataset checksum and code revision.
- Replace Treasury proxy with real overnight curves, market-standard calendars, SOFR compounded cashflows and a multi-curve QuantLib implementation.
- Add realistic bid/ask quotes, lot sizes, collateral and hedge tracking.
- Replace single-split sentiment evaluation with time-stamped news annotations and forward splits.
- Move from lexical retrieval to MiniLM embeddings only after testing whether retrieval quality improves.
- Use constrained deterministic text generation for quantitative facts even if an LLM is added for presentation.
