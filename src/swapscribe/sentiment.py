"""Compact, explainable sentiment model; optional pretrained FinBERT CPU inference."""
from __future__ import annotations
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

LABELS = ["negative", "neutral", "positive"]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Training CSV must have text and label columns")
    out = df[["text", "label"]].dropna().copy()
    out["text"] = out["text"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    out = out[out.label.isin(LABELS) & (out.text.str.len() >= 5)]
    out["normalized"] = out.text.str.lower().str.replace(r"\s+", " ", regex=True)
    conflicts = out.groupby("normalized").label.nunique()
    if (conflicts > 1).any():
        raise ValueError("Identical text with conflicting labels in dataset")
    out = out.drop_duplicates(subset=["normalized"]).drop(columns="normalized")
    if out.label.nunique() < 3 or out.label.value_counts().min() < 3:
        raise ValueError("Need three classes with at least three unique sentences each")
    return out.reset_index(drop=True)


def train_sentiment(data_csv: Path, model_path: Path, report_path: Path, seed=42, dataset_type="demo") -> dict:
    """Hold-out evaluation with deterministic stratified split; no test-set fitting."""
    df = _validate(pd.read_csv(data_csv))
    x_train, x_test, y_train, y_test = train_test_split(
        df.text.to_numpy(), df.label.to_numpy(), test_size=0.25, random_state=seed,
        stratify=df.label.to_numpy(),
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=12_000, sublinear_tf=True),
        LogisticRegression(max_iter=700, class_weight="balanced", random_state=seed),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train.reshape(-1, 1), y_train)
    base_pred = dummy.predict(x_test.reshape(-1, 1))
    report = {
        "dataset_type": dataset_type,
        "data_source": str(data_csv),
        "model_type": "TF-IDF (1-2 grams) + balanced logistic regression",
        "eval_design": "single seeded stratified random 75/25 holdout; unique texts; no temporal information in PhraseBank",
        "warning": ("Synthetic smoke-test score; NOT an external model benchmark" if dataset_type == "demo" else
                    "Financial PhraseBank measures sentence sentiment, not live macro-news performance"),
        "train_count": len(x_train), "test_count": len(x_test),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", labels=LABELS)),
        "baseline_macro_f1": float(f1_score(y_test, base_pred, average="macro", labels=LABELS)),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=LABELS).tolist(),
        "class_report": classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0),
        "seed": seed,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_sentiment_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"No trained sentiment model: {path}. Run `python -m swapscribe.cli train --source demo`.")
    # Load only models produced locally or from a trusted source. joblib/pickle is unsafe for untrusted files.
    return joblib.load(path)


def score_news(model, news: pd.DataFrame) -> pd.DataFrame:
    out = news.copy()
    if out.empty:
        out["sentiment"] = pd.Series(dtype=str)
        out["confidence"] = pd.Series(dtype=float)
        return out
    if "summary" in out.columns and out["summary"].fillna("").str.strip().ne("").any():
        txt = (out.title.fillna("") + ". " + out.summary.fillna("")).astype(str).tolist()
    else:
        txt = out.title.fillna("").astype(str).tolist()
    probabilities = model.predict_proba(txt)
    idx = np.argmax(probabilities, axis=1)
    out["sentiment"] = [str(model.classes_[i]) for i in idx]
    out["confidence"] = probabilities[np.arange(len(out)), idx]
    return out


def finbert_cpu(news: pd.DataFrame) -> pd.DataFrame:
    """Optional ~hundreds-of-MB pretrained model; only called explicitly.

    First call downloads model from Hugging Face. Avoid running both large models
    concurrently on lower-memory laptops. Not needed for the default solution.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install optional: pip install -r requirements-optional-transformers.txt") from exc
    classifier = pipeline("text-classification", model="ProsusAI/finbert", device=-1,
                          truncation=True, max_length=256)
    out = news.copy()
    result = classifier(out.title.fillna("").astype(str).tolist(), batch_size=4)
    out["sentiment"] = [v["label"].lower() for v in result]
    out["confidence"] = [float(v["score"]) for v in result]
    return out
