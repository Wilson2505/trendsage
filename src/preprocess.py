import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_url = re.compile(r"https?://\S+|www\.\S+")
_mention = re.compile(r"@\w+")
_hashtag = re.compile(r"#\w+")
_non_alnum = re.compile(r"[^a-z0-9\s]+")


_analyzer = SentimentIntensityAnalyzer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = _url.sub("", text)
    # Keep the word after # as a token (remove the # only)
    text = _mention.sub("", text)
    text = text.replace("#", " ")
    text = _non_alnum.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    scores = df["text"].fillna("").apply(lambda t: _analyzer.polarity_scores(clean_text(t))["compound"])
    out = df.copy()
    out["sentiment"] = scores
    return out
