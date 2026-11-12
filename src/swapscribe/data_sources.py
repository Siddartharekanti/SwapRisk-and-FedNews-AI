"""Public news RSS and Financial PhraseBank research/education importers."""
from __future__ import annotations
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse
import zipfile
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

FED_RSS = "https://www.federalreserve.gov/feeds/press_all.xml"
FPB_ZIP = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"


def _strip_html(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()[:600]


def parse_rss(xml: str | bytes, origin="Federal Reserve") -> pd.DataFrame:
    if isinstance(xml, str):
        xml = xml.strip().lstrip("\ufeff")
    root = ET.fromstring(xml)
    rows = []
    for item in root.findall(".//item"):
        def field(k):
            return (item.findtext(k) or "").strip()
        raw_dt = field("pubDate")
        try:
            published = parsedate_to_datetime(raw_dt)
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            date = published.date().isoformat()
        except (ValueError, TypeError):
            continue
        link = field("link")
        title = _strip_html(field("title"))
        if not title or not urlparse(link).scheme.startswith("http"):
            continue
        rows.append({"date": date, "title": title, "summary": _strip_html(field("description")),
                     "url": link, "source": origin, "is_synthetic": False})
    return pd.DataFrame(rows, columns=["date", "title", "summary", "url", "source", "is_synthetic"])


def fetch_fed_news(dest: Path, timeout=30) -> Path:
    response = requests.get(FED_RSS, timeout=timeout, headers={"User-Agent": "SwapShield-MacroScribe-Educational/1.0"})
    response.raise_for_status()
    data = parse_rss(response.content)
    if data.empty:
        raise ValueError("No dated Fed RSS items were returned")
    dest.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(dest, index=False)
    return dest


def parse_phrasebank_zip(content: bytes, agreement="75") -> pd.DataFrame:
    """Dataset labels: negative, neutral, positive; no official temporal split."""
    with zipfile.ZipFile(BytesIO(content)) as z:
        options = [n for n in z.namelist() if n.lower().endswith(f"sentences_{agreement}agree.txt")]
        if not options:
            raise ValueError(f"Financial PhraseBank Sentences_{agreement}Agree.txt not found")
        body = z.read(options[0]).decode("latin-1")
    rows = []
    for line in body.splitlines():
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        label = label.lower().strip()
        if label in ("negative", "neutral", "positive") and sentence.strip():
            rows.append((sentence.strip(), label))
    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates(subset="text").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Unusually few Financial PhraseBank examples parsed")
    return df


def fetch_phrasebank(dest: Path, timeout=45, agreement="75") -> Path:
    response = requests.get(FPB_ZIP, timeout=timeout, headers={"User-Agent": "SwapShield-MacroScribe-Educational/1.0"})
    response.raise_for_status()
    df = parse_phrasebank_zip(response.content, agreement=agreement)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest
