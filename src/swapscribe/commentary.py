"""Auditable retrieval and extractive/template-based market commentary.

No uncited quantitative claims. Headlines only provide context; no causal
claims or fabricated evidence. All articles are filtered by observation date.
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    fields = {"date", "title", "summary", "source", "url"}
    if not fields.issubset(df.columns):
        raise ValueError("News CSV missing required columns")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "title"]).drop_duplicates(subset=["date", "title"])
    df["summary"] = df["summary"].fillna("")
    df["url"] = df["url"].fillna("")
    return df.sort_values("date").reset_index(drop=True)


def retrieve(news: pd.DataFrame, query: str, as_of, k=5) -> pd.DataFrame:
    """Sparse semantic-ish retrieval from headlines/summaries, no future leaks."""
    cutoff = pd.Timestamp(as_of)
    available = news[pd.to_datetime(news.date) <= cutoff].copy()
    if available.empty:
        return available.assign(relevance=pd.Series(dtype=float))
    texts = (available.title.fillna("") + ". " + available.summary.fillna("")).astype(str).tolist()
    query = query.strip() or "interest rate treasury yield bond economic monetary policy"
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        matrix = vectorizer.fit_transform(texts + [query])
        similarity = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    except ValueError:
        similarity = np.zeros(len(available), dtype=float)
    available["relevance"] = similarity
    # Recency breaks equal-score ties, rather than presenting irrelevant headlines as strong matches.
    available = available.sort_values(["relevance", "date"], ascending=[False, False]).head(k)
    return available.reset_index(drop=True)


def build_commentary(rates: pd.DataFrame, ranked_news: pd.DataFrame, as_of,
                     sentiment_model=None) -> dict:
    before = rates[pd.to_datetime(rates.date) <= pd.Timestamp(as_of)].sort_values("date")
    if len(before) < 2:
        raise ValueError("At least two dated market observations required for commentary")
    last, prev = before.iloc[-1], before.iloc[-2]
    change_2 = (float(last["2 Yr"]) - float(prev["2 Yr"])) * 100
    change_10 = (float(last["10 Yr"]) - float(prev["10 Yr"])) * 100
    slope = (float(last["10 Yr"]) - float(last["2 Yr"])) * 100
    text = (
        f"As of {pd.Timestamp(last.date).date().isoformat()}, the 2-year U.S. Treasury par yield "
        f"was {float(last['2 Yr']):.2f}% ({change_2:+.1f} bp vs. the preceding available observation), "
        f"while the 10-year yield was {float(last['10 Yr']):.2f}% "
        f"({change_10:+.1f} bp). The 10Y–2Y spread was {slope:+.1f} bp. "
        "These are U.S. Treasury par yields, not SOFR swap rates. "
    )
    news = ranked_news.copy()
    if sentiment_model is not None and not news.empty:
        from .sentiment import score_news
        news = score_news(sentiment_model, news)
    evidence = []
    for _, n in news.iterrows():
        confidence = float(n.confidence) if "confidence" in news.columns else None
        entry = {"date": pd.Timestamp(n.date).date().isoformat(), "title": str(n.title),
                 "source": str(n.source), "url": str(n.url) if pd.notna(n.url) else "",
                 "relevance": round(float(n.relevance), 4) if "relevance" in news.columns else None,
                 "sentiment": str(n.sentiment) if "sentiment" in news.columns else "not_scored",
                 "model_probability": round(confidence, 4) if confidence is not None else None}
        evidence.append(entry)
    if evidence:
        max_rel = max((x["relevance"] or 0.0) for x in evidence)
        if max_rel > 0.05:
            prefix = "Related retrieved headlines (context, not proven causal explanations): "
        else:
            prefix = "Recent Federal Reserve releases for macroeconomic context (context, not proven causal explanations; no high-relevance policy announcement on this observation date): "
        text += (prefix + "; ".join(f"[{i}] {x['title']} ({x['source']}, {x['date']})" for i, x in enumerate(evidence, 1)) + ". ")
    else:
        text += "No source headlines were available on or before the requested cutoff. "
    text += "This is an educational analytics summary, not investment advice."
    return {
        "headline": "U.S. yield-curve market note",
        "requested_as_of": pd.Timestamp(as_of).date().isoformat(),
        "market_observation_date": pd.Timestamp(last.date).date().isoformat(),
        "previous_observation_date": pd.Timestamp(prev.date).date().isoformat(),
        "commentary": text,
        "evidence": evidence,
        "data_type": "Official U.S. Treasury daily par yields; Federal Reserve official releases",
        "limitations": ["No demonstrated causal link between headlines and yield changes",
                        "Research RSS metadata may lag relevant market information",
                        "Point-in-time timing granularity is daily, not intraday"],
    }
