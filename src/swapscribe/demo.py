"""Deterministic, explicitly SYNTHETIC fixtures for offline, end-to-end demos."""
from pathlib import Path
import numpy as np
import pandas as pd

from .market import TENORS

HEADLINES = [
    "Central bank officials discuss the outlook for policy rates",
    "Treasury yields rise ahead of economic data release",
    "Bond traders monitor the latest inflation expectations",
    "Market participants debate the timing of monetary easing",
    "Government bond curve steepens amid shifting term premiums",
    "Retail sales report prompts a reassessment of demand",
    "Economic survey points to moderating manufacturing activity",
    "Fixed income traders focus on an upcoming central bank decision",
    "Inflation readings remain a key driver of rate uncertainty",
    "Investors reassess risks across the bond market",
    "Employment growth data draw attention from rates desks",
    "Central bank statement highlights uncertainty in the outlook",
    "Short-dated Treasury yields change after policy commentary",
    "Financial market volatility rises before economic releases",
    "Bond investors evaluate longer-term inflation risks",
    "Analysts watch the spread between short and long yields",
    "Policy expectations remain sensitive to new macro data",
    "Market commentary reviews the latest yield-curve moves",
]

SENTENCES = {
  "positive": ["Profits increased in the latest reporting period", "Revenue growth exceeded the prior year",
      "Demand improved across the company's markets", "Sales climbed despite challenging conditions",
      "Operating margins expanded during the quarter", "The firm raised its full-year earnings outlook",
      "Cost savings increased overall profitability", "The business reported stronger cash generation",
      "New contract wins supported robust performance", "The bank reported a rise in net income",
      "Investors welcomed better than expected earnings", "The company reduced debt and improved liquidity"],
  "negative": ["Profits declined in the latest reporting period", "Revenue fell below the prior year",
      "Demand weakened across the company's markets", "Sales dropped amid challenging conditions",
      "Operating margins contracted during the quarter", "The firm lowered its full-year earnings outlook",
      "Higher costs reduced overall profitability", "The business reported weaker cash generation",
      "Contract cancellations damaged performance", "The bank reported a fall in net income",
      "Investors reacted to worse than expected earnings", "The company increased debt and worsened liquidity"],
  "neutral": ["The company announced its quarterly reporting date", "The financial institution appointed a new director",
      "The issuer published a statement about its operations", "A scheduled investor presentation will take place",
      "The bank opened a branch in a different city", "The board held a regular meeting this week",
      "The firm provided an update on its business units", "An annual shareholder meeting was announced",
      "The company described the structure of its divisions", "The issuer submitted its annual filing",
      "A conference call was held for investors", "Management provided information about business activities"]
}


def generate_demo(dest: Path, seed=2026) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-08-28", periods=220)
    # Correlated, mean-reverting rate-shock process; entirely synthetic.
    shock = rng.multivariate_normal([0, 0, 0], [[1.9, 1.1, .7], [1.1, 2.5, 1.6], [.7, 1.6, 2.4]], size=len(dates)) / 100
    core = np.array([4.05, 4.15, 4.30]) + np.cumsum(shock, axis=0)
    rows = []
    for idx, d in enumerate(dates):
        x = np.array([2., 5., 10.])
        y = core[idx]
        rates = {col: round(float(np.interp(tenor, x, y, left=y[0]-.08, right=y[-1]+.06)), 4)
                 for col, tenor in TENORS.items()}
        rows.append({"date": d.date().isoformat(), **rates})
    rates_file = dest / "market_rates.csv"
    pd.DataFrame(rows).to_csv(rates_file, index=False)
    portfolio_file = dest / "portfolio.csv"
    pd.DataFrame([
        ["SWAP_A", 9_000_000, 3, 4.10, "pay_fixed"],
        ["SWAP_B", 14_000_000, 4, 3.85, "receive_fixed"],
        ["SWAP_C", 12_000_000, 12, 4.40, "pay_fixed"],
        ["SWAP_D", 7_000_000, 8, 4.05, "pay_fixed"],
    ], columns=["swap_id", "notional_usd", "maturity_years", "fixed_rate_pct", "direction"]).to_csv(portfolio_file, index=False)
    news_rows = []
    for i, headline in enumerate(HEADLINES):
        date = dates[-40+i*2].date().isoformat()
        news_rows.append({"date": date, "title": headline,
                          "summary": "Synthetic training/demo headline, not a historical news report.",
                          "url": "", "source": "SYNTHETIC DEMO", "is_synthetic": True})
    news_file = dest / "news.csv"
    pd.DataFrame(news_rows).to_csv(news_file, index=False)
    train_rows = [{"text": t, "label": label} for label, texts in SENTENCES.items() for t in texts]
    train_file = dest / "sentiment_train.csv"
    pd.DataFrame(train_rows).to_csv(train_file, index=False)
    (dest / "PROVENANCE.txt").write_text(
        "ALL data in data/demo are SYNTHETIC generated fixtures for tests and UI demos.\n"
        "They are NOT U.S. Treasury data, actual client trades, real news, or Financial PhraseBank.\n"
        "Run fetch-market, fetch-news, and fetch-sentiment for separately downloaded public data.\n",
        encoding="utf-8")
    return {"rates": rates_file, "portfolio": portfolio_file, "news": news_file, "sentiment": train_file}
