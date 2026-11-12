from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
import pytest
from swapscribe.data_sources import parse_rss, parse_phrasebank_zip
from swapscribe.commentary import load_news, retrieve, build_commentary
from swapscribe.sentiment import train_sentiment, load_sentiment_model, score_news


def test_rss_parser():
    content = '''<rss><channel><item><title>Rate announcement</title><link>https://www.federalreserve.gov/abc.htm</link>
    <pubDate>Mon, 03 Aug 2026 10:00:00 GMT</pubDate><description>Economic update</description></item></channel></rss>'''
    result = parse_rss(content)
    assert result.iloc[0].date == "2026-08-03"
    assert not bool(result.iloc[0].is_synthetic)


def test_phrasebank_zip_parser():
    data = '\n'.join([f"Company result example {i}@positive" for i in range(40)] +
                     [f"Company outlook example {i}@negative" for i in range(40)] +
                     [f"Company filing example {i}@neutral" for i in range(40)]).encode('latin1')
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('FinancialPhraseBank-v1.0/Sentences_75Agree.txt', data)
    frame = parse_phrasebank_zip(buf.getvalue())
    assert len(frame) == 120
    assert set(frame.label) == {'negative', 'neutral', 'positive'}


def test_future_news_not_retrieved(sample_data):
    news = load_news(sample_data / "news.csv")
    cutoff = news.date.iloc[6]
    result = retrieve(news, "bond rate", cutoff, k=100)
    assert all(result.date <= cutoff)


def test_market_note_is_cited_and_numerically_grounded(rates, sample_data):
    news = load_news(sample_data / "news.csv")
    cutoff = rates.date.iloc[-1]
    ranked = retrieve(news, "rates policy central bank", cutoff, k=3)
    note = build_commentary(rates, ranked, cutoff)
    assert "10Y–2Y spread" in note['commentary']
    assert "not proven causal" in note['commentary']
    assert len(note['evidence']) == 3
    assert all('title' in item and 'source' in item for item in note['evidence'])


def test_training_and_scoring(sample_data, tmp_path):
    model = tmp_path / "model.joblib"
    report_file = tmp_path / "eval.json"
    report = train_sentiment(sample_data / "sentiment_train.csv", model, report_file)
    assert model.exists() and report_file.exists()
    assert 0 <= report["test_macro_f1"] <= 1
    predictor = load_sentiment_model(model)
    texts = pd.DataFrame({"title": ["Profits climbed", "Earnings declined"], "date": ["2025-01-01"] * 2})
    scored = score_news(predictor, texts)
    assert len(scored) == 2
    assert all(0 <= c <= 1 for c in scored.confidence)


def test_training_rejects_conflicting_duplicates(tmp_path):
    file = tmp_path / "bad.csv"
    pd.DataFrame({'text': ['Same headline'] * 2 + ['Good earnings', 'Bad earnings', 'Report published'],
                  'label': ['positive', 'negative', 'positive', 'negative', 'neutral']}).to_csv(file,index=False)
    with pytest.raises(ValueError, match="conflicting"):
        train_sentiment(file, tmp_path / 'm.joblib', tmp_path / 'e.json')
